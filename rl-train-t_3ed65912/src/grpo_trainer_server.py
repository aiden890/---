"""GRPO trainer server (GPU side, xiaomi-cu121 image) for CloseBlenderLid.

Training counterpart of the env card's inference server (rl_server.py). Holds ONE
resident model, freezes the VLM backbone AND the DiT body/projectors, and trains only
PER-SKILL LoRA adapters (src/lora.py) injected into each DiT attention layer, plus an
optional per-skill termination head. This is the operator's per-skill-adapter direction
and -- per the research brief (rl_finetune_brief.md: Flow-GRPO/ReinFlow/piRL/CORAL) --
the memory-safe choice: LoRA optimizer state is a few M params, so AdamW fits (the
full-body probe OOM'd on AdamW; LoRA removes that constraint), though SGD stays default.

GRPO design encodes the brief's guardrails for K=5 flow steps + small group:
  * clipped importance ratio, tight clip (default 0.1),
  * per-group advantage normalisation + advantage clipping to +/-3 sigma,
  * ratio-explosion guard: drop a sample whose ratio leaves [1/ratio_max, ratio_max],
  * light KL-to-reference penalty (behaviour-policy logprob), decayable by the caller,
  * ONE optimizer step per update (no epoch reuse -> low off-policy staleness with SGD),
  * only the ACTIVE skill's adapter is in the graph, so each adapter trains ONLY on its
    own skill segments (strict isolation).

RPC verbs (request["op"]): sample / update / save / load / metrics -- same length-prefixed
pickle framing as the env card servers. "sample" stores the taken states + old logp per
(traj_id, chunk_idx, skill); "update" recomputes with grad under the active skill's
adapter, backprops the GRPO loss, steps, clears the store.
"""
from __future__ import annotations

import argparse
import math
import pickle
import socket
import struct
import sys
import traceback
from pathlib import Path

import torch

_ENV_SRC = Path("/rl_env/src")
sys.path.insert(0, str(_ENV_SRC))
_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
from flow_policy import build_velocity_field  # noqa: E402
from flow_sde import flow_sde_sample, transition_logprob, make_executed_mask  # noqa: E402
# Faithful pi-RL marginal-preserving Flow-SDE (RLinf@bde6c918). Selected via --sampler pirl.
# The default remains the explicitly-named fixed_noise baseline (flow_sde) for backward
# compatibility / byte-identical reproduction of prior runs.
from pirl_flow_sde import (  # noqa: E402
    pirl_flow_sde_sample, pirl_transition_logprob, openpi_timesteps, openpi_sigmas,
    pirl_step_mean_std,
)
from lora import (  # noqa: E402
    SKILLS, inject_per_skill_lora, set_active_skill, lora_parameters, select_trainable,
)
import vlm_scorer  # noqa: E402  (env card: single source of truth for the VQA prompt+math)

# Goal-3 SFT sends DATASET skill names; map them to the per-skill LoRA keys used by
# inject_per_skill_lora (grasp/move_holding/place) -- same roles the GRPO loop's
# SKILL_KEY produces. GRASP_HANDLE->grasp, MOVE_LID_TO_CLOSED->move_holding (lid held
# in transit), RELEASE_HANDLE->place (release/seat). Idempotent: passing a LoRA key or
# None returns it unchanged so the GRPO path and baseline (skill=None) are unaffected.
_SFT_SKILL_TO_LORA = {
    "GRASP_HANDLE": "grasp",
    "MOVE_LID_TO_CLOSED": "move_holding",
    "RELEASE_HANDLE": "place",
}
def _lora_key(skill):
    if skill is None:
        return None
    return _SFT_SKILL_TO_LORA.get(skill, skill)


def assert_no_active_dropout(model):
    """AUDIT FIX #5: fail loudly if any Dropout with p>0 is in TRAIN mode anywhere in the
    policy / action expert. pi-RL disables action-expert dropout so that the sampled
    old-policy log-prob and the recomputed new-policy log-prob use the SAME deterministic
    network; active dropout would make the importance ratio meaningless. Returns the list
    of dropout modules with their (p, training) so the caller can log evidence.
    """
    found = []
    active = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Dropout):
            p = float(getattr(m, "p", 0.0))
            found.append({"name": name, "p": p, "training": bool(m.training)})
            if p > 0.0 and m.training:
                active.append(name)
    if active:
        raise RuntimeError(f"AUDIT #5: {len(active)} active Dropout(p>0) modules during "
                           f"rollout/recompute (first: {active[:3]})")
    return found


class GRPOTrainerServer:
    def __init__(self, a):
        from transformers import AutoModel
        self.a = a
        print("Loading model (GRPO per-skill-LoRA trainer)...", flush=True)
        self.model = AutoModel.from_pretrained(
            a.model, trust_remote_code=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16
        ).cuda().to(torch.bfloat16)
        self.model.eval()
        # AUDIT FIX #5: force every Dropout to eval mode and assert none is active, so the
        # rollout old-logp and the recompute new-logp use the SAME deterministic network
        # (pi-RL disables action-expert dropout; active dropout breaks the importance ratio).
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()
        self.dropout_report = assert_no_active_dropout(self.model)
        targets = tuple(t.strip() for t in a.lora_targets.split(","))
        self.wrappers = inject_per_skill_lora(self.model, skills=SKILLS, rank=a.rank,
                                              alpha=a.alpha, targets=targets)
        n_train, groups = select_trainable(self.model, self.wrappers, a.train_mode)
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"train_mode={a.train_mode} | per-skill LoRA rank={a.rank} into DiT {targets} | "
              f"trainable={n_train/1e6:.2f}M groups={groups} "
              f"({len(self.wrappers)} wrapped x {len(SKILLS)} skills)", flush=True)
        # Arm B (adapter_plus_expert): the action-expert projections are pretrained weights,
        # not zero-init LoRA, so they need a MUCH lower lr than the LoRA to avoid overwriting
        # base competence. Split into two param groups (LoRA at --lr, expert/vlm at --expert-lr).
        expert_lr = getattr(a, "expert_lr", None)
        if a.train_mode != "adapter_only" and expert_lr is not None:
            lora_ps = [p for n, p in self.model.named_parameters() if p.requires_grad and ".lora_" in n]
            extra_ps = [p for n, p in self.model.named_parameters() if p.requires_grad and ".lora_" not in n]
            param_groups = [{"params": lora_ps, "lr": a.lr},
                            {"params": extra_ps, "lr": expert_lr}]
            print(f"param groups: lora lr={a.lr} ({sum(p.numel() for p in lora_ps)/1e6:.2f}M), "
                  f"expert/vlm lr={expert_lr} ({sum(p.numel() for p in extra_ps)/1e6:.2f}M)", flush=True)
        else:
            param_groups = self.trainable_params
        if a.optimizer == "sgd":
            self.opt = torch.optim.SGD(param_groups, lr=a.lr, momentum=0.0)
        elif a.optimizer == "adamw":
            self.opt = torch.optim.AdamW(param_groups, lr=a.lr)
        else:
            raise ValueError(a.optimizer)
        self.store: dict = {}
        print("Model loaded.", flush=True)

    # ---- wire helpers ----
    def _recv_all(self, conn, n):
        data = b""
        while len(data) < n:
            pkt = conn.recv(n - len(data))
            if not pkt:
                return None
            data += pkt
        return data

    def _to_dev(self, v):
        if isinstance(v, torch.Tensor):
            return (v.to(device=self.model.device, dtype=self.model.dtype) if v.is_floating_point()
                    else v.to(device=self.model.device))
        return v

    def _split(self, input_data):
        data = {k: self._to_dev(v) for k, v in input_data.items()}
        state = data.pop("state")
        action_mask = data.pop("action_mask")
        data.pop("task_id", None)
        return state, action_mask, data

    # ---- ops ----
    def op_sample(self, req):
        inputs = req["inputs"]
        eta = float(req.get("eta", self.a.eta))
        skill = req.get("skill")
        state, action_mask, vlm = self._split(inputs)
        # AUDIT FIX #1: adapter activation is decoupled from eta. The trained per-skill
        # LoRA MUST be active for BOTH the deterministic (eta=0) eval rollouts and the
        # stochastic (eta>0) training rollouts -- otherwise before/after eval runs the
        # BASE policy and cannot measure any adapter improvement. Activate whenever a
        # skill is given; pass skill=None only to force the pretrained baseline.
        set_active_skill(self.wrappers, skill)

        # AUDIT FIX #1/#2: derive a UNIQUE, reproducible seed per (episode-seed, chunk).
        # The client sends a stable per-rollout `seed` plus a monotonic `chunk_index`; we
        # combine them so (a) eta=0 eval is fully replayable from a saved x0 seed and (b)
        # successive replan chunks within a trajectory get DIFFERENT noise (the pre-fix
        # bug re-seeded every chunk to the same value -> identical x0/SDE increments).
        base_seed = req.get("seed")
        chunk_index = int(req.get("chunk_index", 0))
        derived_seed = None
        gen = None
        if base_seed is not None:
            derived_seed = (int(base_seed) * 1_000_003 + chunk_index * 9_176 + 1) % (2**31 - 1)
            gen = torch.Generator(device="cuda").manual_seed(derived_seed)

        sampler = getattr(self.a, "sampler", "fixed_noise")
        if eta == 0.0:
            # AUDIT FIX #1: route deterministic eval through the seeded sampler so the
            # initial latent x0 is reproducible (the checkpoint's own forward draws x0 from
            # the global unseeded RNG -> two identical env seeds got different actions). At
            # eta=0 both samplers equal the checkpoint's Euler ODE bit-for-bit, so this
            # changes reproducibility, not behaviour.
            with torch.no_grad():
                vfield, shape, dev, dt = build_velocity_field(self.model, state, action_mask, **vlm)
                if sampler == "pirl":
                    res = pirl_flow_sde_sample(vfield, shape, num_steps=self.a.num_steps,
                                               noise_level=0.0, device=dev,
                                               dtype=action_mask.dtype, generator=gen)
                else:
                    res = flow_sde_sample(vfield, shape, num_steps=self.a.num_steps, eta=0.0, device=dev,
                                          dtype=action_mask.dtype, generator=gen)
            return {"actions": res.actions.cpu(), "logprob": None,
                    "derived_seed": derived_seed, "chunk_index": chunk_index}
        with torch.no_grad():
            vfield, shape, dev, dt = build_velocity_field(self.model, state, action_mask, **vlm)
            exec_mask = make_executed_mask(shape[1], shape[2], self.a.replan_steps,
                                           self.a.real_action_dim, device=dev)
            if sampler == "pirl":
                # --eta maps to the pi-RL exploration noise_level (single knob).
                res = pirl_flow_sde_sample(vfield, shape, num_steps=self.a.num_steps,
                                           noise_level=eta, device=dev,
                                           dtype=action_mask.dtype, generator=gen,
                                           executed_mask=exec_mask)
            else:
                res = flow_sde_sample(vfield, shape, num_steps=self.a.num_steps, eta=eta, device=dev,
                                      dtype=action_mask.dtype, generator=gen, executed_mask=exec_mask)
        old_logp = float(res.executed_logprob().float().cpu()[0])
        chunk = {
            "inputs_cpu": {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()},
            "xs_cpu": [x.detach().cpu() for x in res.xs],
            "exec_mask_cpu": exec_mask.cpu(),
            "old_logp": old_logp,
            "skill": skill,
            "derived_seed": derived_seed,
            "chunk_index": chunk_index,
        }
        # pi-RL uses time-dependent per-step stds; store them so the recompute path uses
        # the IDENTICAL schedule (guarantees ratio==1 on-policy). fixed_noise has constant std.
        if sampler == "pirl":
            chunk["pirl_stds"] = list(res.stds)
        self.store.setdefault(req["traj_id"], []).append(chunk)
        return {"actions": res.actions.cpu(), "logprob": old_logp,
                "derived_seed": derived_seed, "chunk_index": chunk_index}

    def op_update(self, req):
        advantages = req["advantages"]           # {traj_id: advantage float}
        clip = float(req.get("clip", self.a.clip))
        kl_coef = float(req.get("kl_coef", self.a.kl_coef))
        ratio_max = float(req.get("ratio_max", self.a.ratio_max))
        adv_clip = float(req.get("adv_clip", self.a.adv_clip))
        # AUDIT FIX #6: support multiple optimizer epochs over the stored batch so the
        # PPO/GRPO clipping and KL actually engage. Epoch 0 recomputes with the SAME params
        # that generated the rollout, so its ratio==1 exactly (a correctness probe); later
        # epochs move ratio away from 1 and exercise the clip/KL. update_epochs=1 keeps the
        # legacy single-step group-relative REINFORCE behaviour.
        update_epochs = int(req.get("update_epochs", self.a.update_epochs))
        assert_no_active_dropout(self.model)  # AUDIT FIX #5: no stochastic dropout in recompute
        torch.cuda.reset_peak_memory_stats()

        active = [(tid, chunks, max(-adv_clip, min(adv_clip, float(advantages.get(tid, 0.0)))))
                  for tid, chunks in self.store.items() if float(advantages.get(tid, 0.0)) != 0.0]
        epoch_stats = []
        grad_norm = 0.0
        n_chunks_last = 0
        for epoch in range(update_epochs):
            self.opt.zero_grad(set_to_none=True)
            total_loss = 0.0; n_chunks = 0; n_dropped = 0; n_nonfinite = 0
            ratios = []; logratios = []; kl_sum = 0.0; n_clipped = 0
            for traj_id, chunks, adv in active:
                adv_t = torch.tensor(adv, device=self.model.device)
                for chunk in chunks:
                    state, action_mask, vlm = self._split(dict(chunk["inputs_cpu"]))
                    set_active_skill(self.wrappers, chunk["skill"])
                    xs = [x.to(device=self.model.device, dtype=self.model.dtype) for x in chunk["xs_cpu"]]
                    exec_mask = chunk["exec_mask_cpu"].to(self.model.device)
                    old_logp = torch.tensor(chunk["old_logp"], device=self.model.device)
                    vfield, shape, dev, dt = build_velocity_field(self.model, state, action_mask, **vlm)
                    sampler = getattr(self.a, "sampler", "fixed_noise")
                    if sampler == "pirl":
                        # Faithful pi-RL recompute: rebuild per-step means with the SAME
                        # corrected-drift equations and REUSE the stored per-step stds so
                        # epoch-0 ratio==1 exactly. MiBoT velocity at t_m = 1 - t_o.
                        ts = openpi_timesteps(self.a.num_steps)
                        sig = openpi_sigmas(self.a.num_steps, self.a.eta)
                        means = []
                        for k in range(self.a.num_steps):
                            t_o = float(ts[k]); t_next = float(ts[k + 1])
                            t_m = torch.full((shape[0], 1, 1), 1.0 - t_o, device=dev, dtype=action_mask.dtype)
                            if self.a.grad_checkpoint:
                                v = torch.utils.checkpoint.checkpoint(vfield, xs[k], t_m, use_reentrant=False)
                            else:
                                v = vfield(xs[k], t_m)
                            m, _ = pirl_step_mean_std(xs[k], v, t_o, t_next, float(sig[k]))
                            means.append(m)
                        pirl_stds = chunk.get("pirl_stds")
                        new_logp = pirl_transition_logprob(xs, means, pirl_stds,
                                                           executed_mask=exec_mask)[0]
                    else:
                        dtc = 1.0 / self.a.num_steps
                        means = []
                        for k in range(self.a.num_steps):
                            t_k = torch.full((shape[0], 1, 1), k / self.a.num_steps, device=dev, dtype=action_mask.dtype)
                            if self.a.grad_checkpoint:
                                v = torch.utils.checkpoint.checkpoint(vfield, xs[k], t_k, use_reentrant=False)
                            else:
                                v = vfield(xs[k], t_k)
                            means.append(xs[k] + v * dtc)
                        new_logp = transition_logprob(xs, means, eta=self.a.eta,
                                                      num_steps=self.a.num_steps, executed_mask=exec_mask)[0]
                    logratio = new_logp - old_logp
                    if not torch.isfinite(logratio):
                        n_nonfinite += 1
                        continue
                    # AUDIT FIX #6: clamp the log-ratio BEFORE exp so a large excursion cannot
                    # overflow to inf; the ratio-explosion guard then drops the sample.
                    logratio_c = torch.clamp(logratio, math.log(1.0 / ratio_max), math.log(ratio_max))
                    ratio = torch.exp(logratio_c)
                    if not torch.isfinite(ratio) or float(ratio) > ratio_max or float(ratio) < 1.0 / ratio_max:
                        n_dropped += 1
                        continue
                    pg = -torch.min(ratio * adv_t, torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t)
                    kl = (torch.exp(logratio_c) - 1.0) - logratio_c   # KL(new||old) approx
                    loss = pg + kl_coef * kl
                    loss.backward()
                    total_loss += float(pg.detach().cpu())
                    kl_sum += float(kl.detach().cpu())
                    r = float(ratio.detach().cpu())
                    ratios.append(r); logratios.append(float(logratio.detach().cpu()))
                    if abs(r - 1.0) > clip:
                        n_clipped += 1
                    n_chunks += 1
            gn = 0.0
            if n_chunks > 0:
                gn = float(torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=self.a.grad_clip))
                self.opt.step()
            # ESS-like weight diagnostic: (sum w)^2 / sum(w^2), normalized to [0,1]
            ess = None
            if ratios:
                import numpy as _np
                w = _np.asarray(ratios, dtype=_np.float64)
                ess = float((w.sum() ** 2) / ((w ** 2).sum() * len(w))) if (w ** 2).sum() > 0 else None
            epoch_stats.append({
                "epoch": epoch, "loss": total_loss / max(n_chunks, 1), "n_chunks": n_chunks,
                "n_dropped": n_dropped, "n_nonfinite": n_nonfinite,
                "mean_ratio": (sum(ratios) / len(ratios) if ratios else 0.0),
                "mean_abs_logratio": (sum(abs(x) for x in logratios) / len(logratios) if logratios else 0.0),
                "clip_fraction": (n_clipped / n_chunks if n_chunks else 0.0),
                "mean_kl": kl_sum / max(n_chunks, 1), "grad_norm": gn, "ess": ess,
            })
            grad_norm = gn; n_chunks_last = n_chunks
        self.store.clear()
        first, last = epoch_stats[0], epoch_stats[-1]
        return {"loss": last["loss"], "n_chunks": n_chunks_last, "n_dropped": last["n_dropped"],
                "grad_norm": grad_norm, "mean_ratio": last["mean_ratio"], "mean_kl": last["mean_kl"],
                "update_epochs": update_epochs, "epoch0_mean_ratio": first["mean_ratio"],
                "epochLast_mean_ratio": last["mean_ratio"], "epochLast_clip_fraction": last["clip_fraction"],
                "epoch_stats": epoch_stats,
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}

    def op_save(self, req):
        path = Path(req["path"]); path.parent.mkdir(parents=True, exist_ok=True)
        sd = {}
        for i, w in enumerate(self.wrappers):
            for s in w.skills:
                sd[f"w{i}.lora_A.{s}"] = w.lora_A[s].detach().cpu()
                sd[f"w{i}.lora_B.{s}"] = w.lora_B[s].detach().cpu()
        # arms B/C also open non-LoRA params (action expert / vlm slice): save those too
        extra = {n: p.detach().cpu() for n, p in self.model.named_parameters()
                 if p.requires_grad and ".lora_" not in n}
        # Goal-3 reproducibility: persist optimizer + RNG + config + data/model provenance so a
        # resumed/loaded checkpoint is bit-reproducible and auditable (which data, which base rev).
        rng = {
            "torch": torch.get_rng_state(),
            "cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        }
        try:
            import numpy as _np
            rng["numpy"] = _np.random.get_state()
        except Exception:
            pass
        provenance = {
            "data_hash": req.get("data_hash"),          # SHA-256 manifest hashes (from client)
            "data_manifest": req.get("data_manifest"),  # dataset repo/split identity
            "model_path": str(getattr(self.a, "model", "")),
            "model_rev": req.get("model_rev"),          # pinned XiaomiRobotics rev (from client)
            "train_meta": req.get("train_meta"),        # skill/arm/epochs/steps
        }
        torch.save({"lora": sd, "extra_trainable": extra, "config": vars(self.a),
                    "optimizer": self.opt.state_dict(), "rng": rng,
                    "provenance": provenance}, path)
        return {"saved": str(path), "n_lora": len(sd), "n_extra": len(extra),
                "has_optimizer": True, "has_rng": True,
                "data_hash": provenance["data_hash"], "model_rev": provenance["model_rev"]}

    def op_load(self, req):
        blob = torch.load(req["path"], map_location=self.model.device)
        sd = blob["lora"]
        with torch.no_grad():
            for i, w in enumerate(self.wrappers):
                for s in w.skills:
                    w.lora_A[s].copy_(sd[f"w{i}.lora_A.{s}"].to(w.lora_A[s].dtype))
                    w.lora_B[s].copy_(sd[f"w{i}.lora_B.{s}"].to(w.lora_B[s].dtype))
            # AUDIT FIX #2: also restore the non-LoRA trainable slice (arms B/C open the
            # action-expert projections and a VLM slice via extra_trainable). Without this
            # a saved ARM B/C checkpoint loads only its LoRA and silently reverts the
            # expert/VLM weights to pretrained -> a post-hoc eval measures the wrong model.
            extra = blob.get("extra_trainable", {}) or {}
            named = dict(self.model.named_parameters())
            n_extra = 0
            missing = []
            for name, tensor in extra.items():
                if name in named:
                    named[name].copy_(tensor.to(named[name].dtype))
                    n_extra += 1
                else:
                    missing.append(name)
            if missing:
                raise RuntimeError(f"op_load: {len(missing)} extra_trainable params not found "
                                   f"in model (first: {missing[:3]})")
        return {"loaded": req["path"], "n_tensors": len(sd), "n_extra": n_extra}

    def op_metrics(self, req):
        return {"n_trajs": len(self.store), "n_chunks": sum(len(v) for v in self.store.values()),
                "free_gb": round(torch.cuda.mem_get_info()[0] / 1e9, 2)}

    def _yes_no_ids(self):
        """Lazily resolve+cache the yes/no answer token ids from the model's tokenizer."""
        if getattr(self, "_yn_ids", None) is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.a.model, trust_remote_code=True)
            self._yn_ids = vlm_scorer.resolve_yes_no_ids(tok)
        return self._yn_ids

    def op_vlm_score(self, req):
        """Real VQA success/progress score from the policy's own frozen Qwen3-VL backbone.

        Runs one VLM forward over (image + yes/no question) and returns P(yes) read from
        the first-answer-position logits. Auxiliary/diagnostic only -- NEVER a primary
        reward. The VLM is frozen and this runs under no_grad, so it does not perturb any
        adapter/optimizer state (the LoRA wrappers live in the DiT, not the VLM).
        """
        inputs = req["inputs"]
        data = {k: self._to_dev(v) for k, v in inputs.items()}
        yes_ids, no_ids = self._yes_no_ids()
        with torch.no_grad():
            out = self.model.vlm(
                input_ids=data["input_ids"],
                attention_mask=data.get("attention_mask"),
                pixel_values=data.get("pixel_values"),
                image_grid_thw=data.get("image_grid_thw"),
            )
            logits_last = out.logits[0, -1, :]
            prob = vlm_scorer.answer_probability(logits_last, yes_ids, no_ids)
        return {"prob": prob, "question": req.get("question")}

    def op_sft_update(self, req):
        """Goal-3: per-skill conditional flow-matching (CFM) SFT step.

        The client sends, per batch item, the SAME processor conditioning inputs used
        at rollout (3-camera video history + state + skill instruction) plus:
          x1        : target action chunk in the model's action space, [B, L, A_full]
          loss_mask : [B, L, A_full] float mask -- 1.0 only on ACTIVE action dims
                      (std>1e-5, i.e. the 12 real dims) AND action steps INSIDE the
                      skill's [start,end) span; 0.0 elsewhere (the Goal-1 loss mask).
        CFM (rectified-flow) objective, matching the checkpoint's Euler ODE reversed:
          x0 ~ N(0,I); t ~ U(0,1); xt = (1-t)x0 + t*x1; u = x1 - x0
          L  = mean_masked( || v_theta(xt, t) - u ||^2 )
        Only the active per-skill LoRA (or shared LoRA + learned embedding) is trained;
        the backbone stays frozen per train_mode. Supports several optimizer steps.
        """
        skill = _lora_key(req.get("skill"))
        assert skill in (None,) + tuple(SKILLS), f"unknown sft skill {req.get('skill')!r} -> {skill!r}"
        set_active_skill(self.wrappers, skill)
        assert_no_active_dropout(self.model)
        steps = int(req.get("sft_steps", 1))
        seed = req.get("seed")
        gen = torch.Generator(device="cuda").manual_seed(int(seed)) if seed is not None else None
        inputs = req["inputs"]
        state, action_mask, vlm = self._split(inputs)
        x1 = self._to_dev(req["x1"])                         # [B,L,A_full]
        loss_mask = self._to_dev(req["loss_mask"]).to(self.model.dtype)  # [B,L,A_full]
        # REDESIGN (operator decision A, run30): KL-to-base anchor to stop the per-skill
        # adapter from overwriting the pretrained full-task competence. On the SAME (xt,t)
        # demo-derived draw, the BASE velocity (adapter OFF) v_base is the pretrained
        # policy's flow field; anchoring v_theta -> v_base in action/velocity space is the
        # exact tractable KL surrogate for this fixed-noise Gaussian flow transition
        # (mu = xt + v*dt, sigma const): KL(N(v_theta) || N(v_base)) = ||v_theta - v_base||^2/(2 sig^2).
        # So loss = masked_mse(v_theta, u_demo) + anchor_coef * masked_mse(v_theta, v_base).
        # anchor_coef=0 reproduces the prior pure-BC objective bit-for-bit.
        anchor_coef = float(req.get("anchor_coef", getattr(self.a, "anchor_coef", 0.0)))

        losses = []
        bc_losses = []
        anc_losses = []
        gn = 0.0
        for _ in range(steps):
            self.opt.zero_grad(set_to_none=True)
            vfield, shape, dev, dt = build_velocity_field(self.model, state, action_mask, **vlm)
            x0 = torch.randn(shape, device=dev, dtype=action_mask.dtype, generator=gen)
            t = torch.rand((shape[0], 1, 1), device=dev, dtype=action_mask.dtype, generator=gen)
            xt = (1.0 - t) * x0 + t * x1
            u = x1 - x0
            denom = loss_mask.sum().clamp_min(1.0)
            # base velocity at the same (xt,t) with the adapter disabled (no grad) -- shares
            # the already-encoded VLM/state conditioning from the same vfield closure.
            v_base = None
            if anchor_coef > 0.0:
                set_active_skill(self.wrappers, None)
                with torch.no_grad():
                    v_base = vfield(xt, t).detach()
                set_active_skill(self.wrappers, skill)
            if self.a.grad_checkpoint:
                v = torch.utils.checkpoint.checkpoint(vfield, xt, t, use_reentrant=False)
            else:
                v = vfield(xt, t)
            bc = ((v - u) ** 2 * loss_mask).sum() / denom
            if anchor_coef > 0.0:
                anc = ((v - v_base) ** 2 * loss_mask).sum() / denom
                loss = bc + anchor_coef * anc
                anc_losses.append(float(anc.detach().cpu()))
            else:
                loss = bc
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=self.a.grad_clip))
            self.opt.step()
            losses.append(float(loss.detach().cpu()))
            bc_losses.append(float(bc.detach().cpu()))
        return {"loss": losses[-1], "loss_first": losses[0], "losses": losses,
                "bc_loss": bc_losses[-1], "anchor_loss": (anc_losses[-1] if anc_losses else 0.0),
                "anchor_coef": anchor_coef,
                "grad_norm": gn, "n_active": int(loss_mask.sum().item()),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}

    def op_sft_val(self, req):
        """CFM validation loss (no grad, no optimizer step). Averages over `mc` random
        (x0,t) draws to reduce Monte-Carlo variance of the flow-matching loss."""
        skill = _lora_key(req.get("skill"))
        assert skill in (None,) + tuple(SKILLS), f"unknown sft skill {req.get('skill')!r} -> {skill!r}"
        set_active_skill(self.wrappers, skill)
        mc = int(req.get("mc", 8))
        seed = req.get("seed")
        gen = torch.Generator(device="cuda").manual_seed(int(seed)) if seed is not None else None
        state, action_mask, vlm = self._split(req["inputs"])
        x1 = self._to_dev(req["x1"])
        loss_mask = self._to_dev(req["loss_mask"]).to(self.model.dtype)
        with torch.no_grad():
            vfield, shape, dev, dt = build_velocity_field(self.model, state, action_mask, **vlm)
            tot = 0.0
            for _ in range(mc):
                x0 = torch.randn(shape, device=dev, dtype=action_mask.dtype, generator=gen)
                t = torch.rand((shape[0], 1, 1), device=dev, dtype=action_mask.dtype, generator=gen)
                xt = (1.0 - t) * x0 + t * x1
                u = x1 - x0
                v = vfield(xt, t)
                se = (v - u) ** 2 * loss_mask
                tot += float((se.sum() / loss_mask.sum().clamp_min(1.0)).cpu())
        return {"val_loss": tot / mc, "mc": mc, "n_active": int(loss_mask.sum().item())}

    def handle(self, req):
        fn = {"sample": self.op_sample, "update": self.op_update, "save": self.op_save,
              "load": self.op_load, "metrics": self.op_metrics,
              "vlm_score": self.op_vlm_score,
              "sft_update": self.op_sft_update, "sft_val": self.op_sft_val}.get(req.get("op"))
        return fn(req) if fn else {"error": f"unknown op {req.get('op')}"}

    def serve(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.a.host, self.a.port))
            s.listen(1)
            print(f"GRPO trainer server on {self.a.host}:{self.a.port}...", flush=True)
            while True:
                conn, _ = s.accept()
                try:
                    while True:
                        ln = self._recv_all(conn, 4)
                        if not ln:
                            break
                        n = struct.unpack(">I", ln)[0]
                        payload = self._recv_all(conn, n)
                        if not payload:
                            break
                        resp = self.handle(pickle.loads(payload))
                        out = pickle.dumps(resp)
                        conn.sendall(struct.pack(">I", len(out)) + out)
                except Exception as e:
                    print(f"Error: {e}", flush=True)
                    traceback.print_exc()
                    try:
                        out = pickle.dumps({"error": str(e)})
                        conn.sendall(struct.pack(">I", len(out)) + out)
                    except Exception:
                        pass
                finally:
                    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/checkpoint")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=10088)
    ap.add_argument("--lr", type=float, default=2e-3)          # SGD LoRA (brief: 1e-3..5e-3)
    ap.add_argument("--optimizer", default="sgd")
    ap.add_argument("--train-mode", default="adapter_only",
                    choices=("adapter_only", "adapter_plus_expert", "adapter_plus_expert_vlm"))
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--expert-lr", type=float, default=None,
                    help="arm B/C: separate (much lower) lr for the pretrained action-expert/VLM "
                         "slice; LoRA keeps --lr. None = single lr for all trainables.")
    ap.add_argument("--anchor-coef", type=float, default=0.0,
                    help="Goal-3 SFT KL-to-base anchor weight: adds anchor_coef*||v_theta-v_base||^2 "
                         "(base = adapter-off pretrained velocity) to the CFM loss to preserve base "
                         "competence. 0 = pure BC (prior behaviour).")
    ap.add_argument("--lora-targets", default="qkv_proj")
    ap.add_argument("--num-steps", type=int, default=5)
    ap.add_argument("--sampler", default="fixed_noise", choices=["fixed_noise", "pirl"],
                    help="RL rollout sampler. 'fixed_noise' = the explicitly-named baseline "
                         "(constant sigma=eta*sqrt(dt), uncorrected drift; NOT marginal-preserving). "
                         "'pirl' = faithful pi-RL marginal-preserving Flow-SDE (RLinf@bde6c918). "
                         "Do NOT label the fixed_noise sampler pi-RL/Flow-SDE.")
    ap.add_argument("--eta", type=float, default=0.6,
                    help="Exploration noise. For fixed_noise: constant sigma=eta*sqrt(dt). "
                         "For pirl: the noise_level in sigma=noise_level*sqrt(t/(1-t)).")
    ap.add_argument("--replan-steps", type=int, default=16)
    ap.add_argument("--real-action-dim", type=int, default=12)
    ap.add_argument("--clip", type=float, default=0.1)         # tighter than 0.2 (K=5 product)
    ap.add_argument("--kl-coef", type=float, default=0.005)
    ap.add_argument("--ratio-max", type=float, default=10.0)
    ap.add_argument("--adv-clip", type=float, default=3.0)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--update-epochs", type=int, default=1,
                    help="PPO/GRPO optimizer epochs over each stored rollout batch. 1 = "
                         "single-step group-relative REINFORCE (ratio==1, no clipping); "
                         ">1 exercises the clipped objective + KL (pi-RL uses 4).")
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    ap.set_defaults(grad_checkpoint=True)
    a = ap.parse_args()
    GRPOTrainerServer(a).serve()


if __name__ == "__main__":
    main()
