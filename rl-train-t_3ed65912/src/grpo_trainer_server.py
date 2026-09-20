"""GRPO trainer server (GPU side, xiaomi-cu121 image) for CloseBlenderLid.

Training counterpart of the env card's inference server (rl_server.py). Holds ONE
resident model, freezes the VLM backbone AND the DiT body/projectors, and trains only
PER-SKILL LoRA adapters (src/lora.py) injected into configured action-expert Linear layers, plus an
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
import copy
import hashlib
import math
import os
import pickle
import random
import socket
import struct
import sys
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
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
    pirl_flow_sde_sample, pirl_transition_logprob_elements, openpi_timesteps, openpi_sigmas,
    pirl_step_mean_std,
)
from lora import (  # noqa: E402
    SKILLS, inject_per_skill_lora, load_lora_state_dict, lora_state_dict,
    set_active_skill, lora_parameters, select_trainable,
)
from training_correctness import (  # noqa: E402
    mean_loss_scale, validate_optimizer_config,
    verify_deployment_manifest, should_stop_for_kl,
)
from checkpoint_schema import (  # noqa: E402
    build_checkpoint_metadata, rng_state_for_restore, validate_checkpoint_metadata,
)
from update_batch import IdempotentUpdateCache  # noqa: E402
try:
    from rollout_store import export_rollout_store, import_rollout_store  # noqa: E402
except ImportError:  # Optional unless the RLinf collector integration is mounted.
    export_rollout_store = import_rollout_store = None

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
        targets = tuple(t.strip() for t in a.lora_targets.split(",") if t.strip())
        adapter_skills = tuple(s.strip() for s in a.adapter_skills.split(",") if s.strip())
        if not adapter_skills or len(set(adapter_skills)) != len(adapter_skills):
            raise ValueError(f"--adapter-skills must be non-empty and unique: {adapter_skills}")
        unknown_skills = sorted(set(adapter_skills) - set(SKILLS))
        if unknown_skills:
            raise ValueError(f"unknown --adapter-skills: {unknown_skills}")
        self.adapter_skills = adapter_skills
        self.wrappers = inject_per_skill_lora(self.model, skills=adapter_skills, rank=a.rank,
                                              alpha=a.alpha, targets=targets)
        if not self.wrappers:
            raise RuntimeError(f"LoRA target selection matched no Linear modules: {targets}")
        self.lora_target_modules = [w.target_name for w in self.wrappers]
        n_train, groups = select_trainable(self.model, self.wrappers, a.train_mode)
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"train_mode={a.train_mode} | per-skill LoRA rank={a.rank} into DiT {targets} | "
              f"trainable={n_train/1e6:.2f}M groups={groups} "
              f"({len(self.wrappers)} wrapped x {len(adapter_skills)} skills)", flush=True)
        print("LoRA target modules:\n  " + "\n  ".join(self.lora_target_modules), flush=True)
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
            self.opt = torch.optim.AdamW(param_groups, lr=a.lr, weight_decay=a.weight_decay)
        else:
            raise ValueError(a.optimizer)
        self.store: dict = {}
        self.update_cache = IdempotentUpdateCache()
        # One resident model serves multiple simulator clients. Connections may overlap,
        # while model/store mutations remain serialized through this actor lock.
        self.request_lock = threading.RLock()
        self.policy_version = 0
        self.policy_hash = self._compute_policy_hash()
        # A version is an immutable name for one exact set of mutable policy tensors.
        # Retain the current and immediately preceding identities so lag-1 rollout
        # payloads can be authenticated, rather than accepting any caller-provided hash.
        self.policy_hash_history = {self.policy_version: self.policy_hash}
        self.fatal_error = None
        print("Model loaded.", flush=True)

    def _compute_policy_hash(self):
        """Fingerprint all mutable policy tensors without serializing optimizer state."""
        digest = hashlib.sha256()
        for name, parameter in sorted(self.model.named_parameters()):
            if not parameter.requires_grad:
                continue
            value = parameter.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _remember_policy_identity(self, version, policy_hash):
        version, policy_hash = int(version), str(policy_hash)
        known = self.policy_hash_history.get(version)
        if known is not None and known != policy_hash:
            raise RuntimeError(
                f"policy version v{version} is immutable: known={known}, incoming={policy_hash}")
        self.policy_hash_history[version] = policy_hash
        floor = self.policy_version - 1
        self.policy_hash_history = {
            key: value for key, value in self.policy_hash_history.items() if key >= floor
        }

    @staticmethod
    def _cpu_clone(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: GRPOTrainerServer._cpu_clone(item) for key, item in value.items()}
        if isinstance(value, list):
            return [GRPOTrainerServer._cpu_clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(GRPOTrainerServer._cpu_clone(item) for item in value)
        return copy.deepcopy(value)

    def _snapshot_mutable_state(self, *, include_optimizer):
        snapshot = {
            "parameters": [parameter.detach().cpu().clone()
                           for parameter in self.trainable_params],
            "policy_version": self.policy_version,
            "policy_hash": self.policy_hash,
            "policy_hash_history": dict(self.policy_hash_history),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            "python_rng": random.getstate(),
        }
        try:
            import numpy as _np
            snapshot["numpy_rng"] = _np.random.get_state()
        except Exception:
            pass
        if include_optimizer:
            snapshot["optimizer"] = self._cpu_clone(self.opt.state_dict())
        return snapshot

    def _restore_mutable_state(self, snapshot):
        with torch.no_grad():
            if len(snapshot["parameters"]) != len(self.trainable_params):
                raise RuntimeError("rollback trainable-parameter layout changed")
            for parameter, saved in zip(self.trainable_params, snapshot["parameters"]):
                if tuple(parameter.shape) != tuple(saved.shape):
                    raise RuntimeError("rollback trainable-parameter shape changed")
                parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))
        if "optimizer" in snapshot:
            self.opt.load_state_dict(snapshot["optimizer"])
        self.policy_version = int(snapshot["policy_version"])
        self.policy_hash = str(snapshot["policy_hash"])
        self.policy_hash_history = dict(snapshot["policy_hash_history"])
        torch.set_rng_state(snapshot["torch_rng"])
        if torch.cuda.is_available() and snapshot.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(snapshot["cuda_rng"])
        random.setstate(snapshot["python_rng"])
        if "numpy_rng" in snapshot:
            import numpy as _np
            _np.random.set_state(snapshot["numpy_rng"])
        actual = self._compute_policy_hash()
        if actual != self.policy_hash:
            raise RuntimeError(
                f"rollback policy hash mismatch: expected={self.policy_hash}, actual={actual}")

    def _rollback_or_fail_stop(self, snapshot, cause):
        try:
            self._restore_mutable_state(snapshot)
        except Exception as rollback_error:
            self.fatal_error = (
                f"policy transaction failed ({cause!r}) and rollback failed "
                f"({rollback_error!r}); server is fail-stopped")
            raise RuntimeError(self.fatal_error) from rollback_error

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

    def _store_inputs_cpu(self, inputs):
        """Store exactly the tensor precision consumed by the policy.

        Processor video tensors arrive as float32, but ``_to_dev`` always casts every
        floating input to the BF16 model dtype before the first policy forward.  Keeping
        the unused float32 mantissa in every rollout chunk roughly doubles both the
        serialized payload and DAX-to-learner traffic.  Casting before the CPU copy is
        therefore lossless with respect to the model-visible input.
        """
        return {
            key: (value.detach().to(device="cpu", dtype=self.model.dtype)
                  if isinstance(value, torch.Tensor) and value.is_floating_point()
                  else value.detach().cpu() if isinstance(value, torch.Tensor)
                  else value)
            for key, value in inputs.items()
        }

    def _split(self, input_data):
        data = {k: self._to_dev(v) for k, v in input_data.items()}
        state = data.pop("state")
        action_mask = data.pop("action_mask")
        data.pop("task_id", None)
        return state, action_mask, data

    # ---- ops ----
    def op_sample(self, req):
        expected_version = req.get("expected_policy_version")
        if expected_version is not None and int(expected_version) != self.policy_version:
            raise RuntimeError(
                f"actor policy changed before episode completed: expected v{expected_version}, "
                f"active v{self.policy_version}")
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
                    "derived_seed": derived_seed, "chunk_index": chunk_index,
                    "policy_version": self.policy_version, "policy_hash": self.policy_hash}
        with torch.no_grad():
            vfield, shape, dev, dt = build_velocity_field(self.model, state, action_mask, **vlm)
            exec_mask = make_executed_mask(shape[1], shape[2], self.a.replan_steps,
                                           self.a.real_action_dim, device=dev)
            denoise_index = None
            if sampler == "pirl":
                # --eta maps to the pi-RL exploration noise_level (single knob).
                # RLinf Flow-SDE uses joint_logprob=False: one randomly selected denoise
                # transition is stochastic; all other denoise steps follow the ODE.
                denoise_index = (int(derived_seed) % self.a.num_steps
                                 if derived_seed is not None else chunk_index % self.a.num_steps)
                res = pirl_flow_sde_sample(vfield, shape, num_steps=self.a.num_steps,
                                           noise_level=eta, device=dev,
                                           dtype=action_mask.dtype, generator=gen,
                                           executed_mask=exec_mask,
                                           denoise_index=denoise_index)
            else:
                res = flow_sde_sample(vfield, shape, num_steps=self.a.num_steps, eta=eta, device=dev,
                                      dtype=action_mask.dtype, generator=gen, executed_mask=exec_mask)
        old_logp = float(res.executed_logprob().float().cpu()[0])
        chunk = {
            "inputs_cpu": self._store_inputs_cpu(inputs),
            "xs_cpu": [x.detach().cpu() for x in res.xs],
            "exec_mask_cpu": exec_mask.cpu(),
            "old_logp": old_logp,
            "skill": skill,
            # Per-trajectory sampler value: one resident actor serves a parameter grid, so
            # recompute must not silently substitute the server's default --eta.
            "eta": eta,
            "derived_seed": derived_seed,
            "chunk_index": chunk_index,
            "policy_version": self.policy_version,
            "policy_hash": self.policy_hash,
        }
        # pi-RL uses time-dependent per-step stds; store them so the recompute path uses
        # the IDENTICAL schedule (guarantees ratio==1 on-policy). fixed_noise has constant std.
        if sampler == "pirl":
            chunk["pirl_stds"] = list(res.stds)
            chunk["denoise_index"] = denoise_index
            chunk["old_logp_elements_cpu"] = (
                res.perstep_perdim_logprob[0, denoise_index].detach().cpu())
        self.store.setdefault(req["traj_id"], []).append(chunk)
        return {"actions": res.actions.cpu(), "logprob": old_logp,
                "derived_seed": derived_seed, "chunk_index": chunk_index,
                "policy_version": self.policy_version, "policy_hash": self.policy_hash}

    def _forward_new_logp_terms(self, chunk):
        """Recompute CURRENT-policy ratio terms for one stored rollout chunk.

        Single source of truth used by BOTH the optimizer
        epoch loop (grad flows) and the post-step diagnostic pass (under no_grad). Keeping
        one implementation guarantees the epoch-0 ratio==1 correctness probe and the
        post-step ratio use identical math. For pi-RL Flow-SDE these are independent
        [executed timestep, action-dimension] terms from ONE selected denoise transition,
        matching RLinf's ``joint_logprob=False`` path. MiBoT time is t_m = 1 - t_o.
        """
        state, action_mask, vlm = self._split(dict(chunk["inputs_cpu"]))
        set_active_skill(self.wrappers, chunk["skill"])
        xs = [x.to(device=self.model.device, dtype=self.model.dtype) for x in chunk["xs_cpu"]]
        exec_mask = chunk["exec_mask_cpu"].to(self.model.device)
        vfield, shape, dev, dt = build_velocity_field(self.model, state, action_mask, **vlm)
        sampler = getattr(self.a, "sampler", "fixed_noise")
        chunk_eta = float(chunk.get("eta", self.a.eta))
        if sampler == "pirl":
            ts = openpi_timesteps(self.a.num_steps)
            sig = openpi_sigmas(self.a.num_steps, chunk_eta)
            selected = int(chunk["denoise_index"])
            # joint_logprob=False uses exactly one stochastic transition.  The prior
            # implementation recomputed all N DiT means even though
            # pirl_transition_logprob_elements reads only means[selected].  Computing
            # the selected mean alone is algebraically identical and removes N-1
            # unused gradient forwards (and checkpoint recomputations) per chunk.
            t_o = float(ts[selected]); t_next = float(ts[selected + 1])
            t_m = torch.full(
                (shape[0], 1, 1), 1.0 - t_o, device=dev, dtype=action_mask.dtype)
            if self.a.grad_checkpoint:
                v = torch.utils.checkpoint.checkpoint(
                    vfield, xs[selected], t_m, use_reentrant=False)
            else:
                v = vfield(xs[selected], t_m)
            selected_mean, _ = pirl_step_mean_std(
                xs[selected], v, t_o, t_next, float(sig[selected]))
            means = [None] * self.a.num_steps
            means[selected] = selected_mean
            per = pirl_transition_logprob_elements(
                xs, means, chunk["pirl_stds"], selected)[0]
            return per[exec_mask]
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
            new_logp = transition_logprob(xs, means, eta=chunk_eta,
                                          num_steps=self.a.num_steps, executed_mask=exec_mask)[0]
            return new_logp.reshape(1)

    def _old_logp_terms(self, chunk):
        if getattr(self.a, "sampler", "fixed_noise") == "pirl":
            old = chunk.get("old_logp_elements_cpu")
            if old is None:
                raise RuntimeError("pi-RL rollout is missing independent old log-prob terms")
            mask = chunk["exec_mask_cpu"].to(torch.bool)
            return old[mask].to(device=self.model.device, dtype=torch.float32)
        return torch.tensor([chunk["old_logp"]], device=self.model.device, dtype=torch.float32)

    @staticmethod
    def _chunk_batch_key(chunk):
        """Compatibility key for exact pi-RL microbatching."""
        inputs = chunk["inputs_cpu"]
        tensor_layout = tuple(sorted(
            (key, tuple(value.shape[1:]), str(value.dtype))
            for key, value in inputs.items() if isinstance(value, torch.Tensor)))
        constants = tuple(sorted(
            (key, repr(value)) for key, value in inputs.items()
            if not isinstance(value, torch.Tensor)))
        return (chunk.get("skill"), int(chunk.get("denoise_index", -1)),
                float(chunk.get("eta", 0.0)), tensor_layout, constants)

    def _microbatches(self, items):
        size = max(1, int(getattr(self.a, "microbatch_size", 1)))
        pending = {}
        ready = []
        for item in items:
            key = self._chunk_batch_key(item[1])
            bucket = pending.setdefault(key, [])
            bucket.append(item)
            if len(bucket) == size:
                ready.append(bucket[:])
                bucket.clear()
        ready.extend(bucket for bucket in pending.values() if bucket)
        return ready

    @staticmethod
    def _collate_inputs(chunks):
        keys = tuple(chunks[0]["inputs_cpu"])
        if any(tuple(chunk["inputs_cpu"]) != keys for chunk in chunks):
            raise ValueError("microbatch input keys differ")
        collated = {}
        for key in keys:
            values = [chunk["inputs_cpu"][key] for chunk in chunks]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.cat(values, dim=0)
            else:
                if any(value != values[0] for value in values[1:]):
                    raise ValueError(f"microbatch constant input differs: {key}")
                collated[key] = values[0]
        return collated

    def _forward_new_logp_terms_batch(self, chunks):
        """Return one flattened term tensor per chunk, sharing one batched VLM/DiT pass."""
        if (len(chunks) == 1
                or getattr(self.a, "sampler", "fixed_noise") != "pirl"
                or not bool(getattr(self.a, "batch_model_forward", False))):
            return [self._forward_new_logp_terms(chunk) for chunk in chunks]
        keys = {self._chunk_batch_key(chunk) for chunk in chunks}
        if len(keys) != 1:
            raise ValueError("incompatible pi-RL chunks in one microbatch")
        state, action_mask, vlm = self._split(self._collate_inputs(chunks))
        set_active_skill(self.wrappers, chunks[0]["skill"])
        xs = [torch.cat([
            chunk["xs_cpu"][index].to(device=self.model.device, dtype=self.model.dtype)
            for chunk in chunks], dim=0) for index in range(self.a.num_steps + 1)]
        masks = [chunk["exec_mask_cpu"].to(self.model.device) for chunk in chunks]
        vfield, shape, dev, _ = build_velocity_field(
            self.model, state, action_mask, **vlm)
        selected = int(chunks[0]["denoise_index"])
        chunk_eta = float(chunks[0].get("eta", self.a.eta))
        ts = openpi_timesteps(self.a.num_steps)
        sig = openpi_sigmas(self.a.num_steps, chunk_eta)
        t_o = float(ts[selected]); t_next = float(ts[selected + 1])
        t_m = torch.full(
            (shape[0], 1, 1), 1.0 - t_o, device=dev, dtype=action_mask.dtype)
        if self.a.grad_checkpoint:
            v = torch.utils.checkpoint.checkpoint(
                vfield, xs[selected], t_m, use_reentrant=False)
        else:
            v = vfield(xs[selected], t_m)
        selected_mean, _ = pirl_step_mean_std(
            xs[selected], v, t_o, t_next, float(sig[selected]))
        means = [None] * self.a.num_steps
        means[selected] = selected_mean
        per = pirl_transition_logprob_elements(
            xs, means, chunks[0]["pirl_stds"], selected)
        return [per[index][masks[index]] for index in range(len(chunks))]

    def _adapter_vector(self):
        """Flattened detached copy of every trainable parameter (for the adapter-delta metric)."""
        with torch.no_grad():
            return torch.cat([p.detach().float().flatten() for p in self.trainable_params]).clone()

    def _post_step_diagnostics(self, active, clip, ratio_max):
        """AUDIT/measurement fix: recompute ratio/approx-KL/clip-fraction/ESS AFTER the
        optimizer step, under no_grad. With update_epochs=1 the epoch-0 metrics are the
        pre-step correctness probe (ratio==1, KL==0) and say NOTHING about update strength;
        THIS pass measures how far the updated policy actually moved off the rollout policy.
        """
        import numpy as _np
        flat = [(traj_id, chunk, adv) for traj_id, chunks, adv in active for chunk in chunks]
        total_chunks = len(flat)
        limit = max(1, int(getattr(self.a, "post_diag_chunks", 128)))
        if total_chunks > limit:
            indices = [(index * total_chunks) // limit for index in range(limit)]
            sampled = [flat[index] for index in indices]
        else:
            sampled = flat
        collected = []; collected_sizes = []
        max_logratio = math.log(ratio_max)
        with torch.no_grad():
            for microbatch in self._microbatches(sampled):
                chunks = [item[1] for item in microbatch]
                new_batch = self._forward_new_logp_terms_batch(chunks)
                for (_, chunk, _), new_terms in zip(microbatch, new_batch):
                    old_terms = self._old_logp_terms(chunk)
                    values = (new_terms - old_terms).detach().float()
                    collected.append(values)
                    collected_sizes.append(values.numel())
        raw = (torch.cat(collected).cpu().numpy() if collected
               else _np.asarray([], dtype=_np.float32))
        finite_mask = _np.isfinite(raw)
        finite = raw[finite_mask]
        bounded = _np.clip(finite, -max_logratio, max_logratio)
        ratios = _np.exp(bounded)
        n_nonfinite = int((~finite_mask).sum())
        n_clamped = int((bounded != finite).sum())
        n_clipped = int((_np.abs(ratios - 1.0) > clip).sum())
        kl_sum = float(((ratios - 1.0) - bounded).sum())
        n_dropped = 0; offset = 0
        for size in collected_sizes:
            n_dropped += int(not finite_mask[offset:offset + size].any())
            offset += size
        ess = None
        if ratios.size:
            w = ratios.astype(_np.float64, copy=False)
            ess = float((w.sum() ** 2) / ((w ** 2).sum() * len(w))) if (w ** 2).sum() > 0 else None
        return {
            "post_step_mean_ratio": (float(ratios.mean()) if ratios.size else None),
            "post_step_mean_abs_ratio_dev": (
                float(_np.abs(ratios - 1.0).mean()) if ratios.size else None),
            "post_step_clip_fraction": (n_clipped / len(ratios) if ratios.size else None),
            "post_step_mean_kl": (kl_sum / len(ratios) if ratios.size else None),
            "post_step_ess": ess, "post_step_n_chunks": len(sampled),
            "post_step_total_chunks": total_chunks,
            "post_step_sampling": "evenly_spaced" if len(sampled) < total_chunks else "all",
            "post_step_n_ratio_terms": len(ratios),
            "post_step_n_nonfinite": n_nonfinite,
            "post_step_n_dropped": n_dropped,
            "post_step_n_clamped": n_clamped,
        }

    def op_config(self, req):
        """Full trainer configuration for the run summary (optimizer/LR/schedule/grad-clip/
        weight-decay/LoRA rank-alpha-targets/trainable count/sampler/eta/code rev). The
        client logs this verbatim so every run summary is self-describing (measurement fix).
        """
        cfg = vars(self.a)
        n_train = sum(p.numel() for p in self.trainable_params)
        opt_groups = []
        for g in self.opt.param_groups:
            opt_groups.append({"lr": g.get("lr"), "weight_decay": g.get("weight_decay"),
                               "n_params": sum(p.numel() for p in g["params"])})
        total_params = sum(p.numel() for p in self.model.parameters())
        return {"config": cfg, "trainable_params": int(n_train),
                "total_model_params": int(total_params),
                "trainable_fraction": float(n_train / total_params),
                "lora_target_modules": self.lora_target_modules,
                "optimizer": type(self.opt).__name__, "optimizer_groups": opt_groups,
                "code_rev": req.get("code_rev"), "dropout_report": self.dropout_report}

    def op_update(self, req):
        """Update atomically with respect to the rollout store, including failures."""
        update_id = req.get("update_id")
        if update_id in self.update_cache.completed:
            # A client may reconnect after the optimizer committed but its response was
            # lost. Drop any replayed rollout store and return the original result.
            self.store.clear()
            replay = self.update_cache.run(update_id, lambda: None)
            replay["replayed_update"] = True
            return replay

        def perform():
            transaction = self._snapshot_mutable_state(include_optimizer=True)
            try:
                snapshots = {(chunk.get("policy_version", self.policy_version),
                              chunk.get("policy_hash", self.policy_hash))
                             for chunks in self.store.values() for chunk in chunks}
                if len(snapshots) > 1:
                    raise ValueError(f"one update cannot mix policy snapshots: {sorted(snapshots)}")
                rollout_version = next(iter(snapshots))[0] if snapshots else self.policy_version
                rollout_hash = (next(iter(snapshots))[1] if snapshots else self.policy_hash)
                lag = self.policy_version - int(rollout_version)
                if lag < 0 or lag > 1:
                    raise ValueError(
                        f"rollout policy lag must be 0 or 1: rollout=v{rollout_version}, "
                        f"learner=v{self.policy_version}")
                expected_hash = self.policy_hash_history.get(int(rollout_version))
                if expected_hash is None or rollout_hash != expected_hash:
                    raise ValueError(
                        f"rollout policy identity is not retained: v{rollout_version}/"
                        f"{rollout_hash}, expected hash={expected_hash}")
                version_before = self.policy_version
                result = self._op_update_impl(req)
                if float(result.get("adapter_delta_l2", 0.0)) > 0.0:
                    self.policy_version += 1
                    self.policy_hash = self._compute_policy_hash()
                    self._remember_policy_identity(self.policy_version, self.policy_hash)
                result["rollout_policy_version"] = int(rollout_version)
                result["policy_lag"] = lag
                result["policy_version_before"] = version_before
                result["policy_version_after"] = self.policy_version
                result["policy_hash_after"] = self.policy_hash
                result["replayed_update"] = False
                result["curriculum_state"] = req.get("curriculum_state")
                return result
            except Exception as error:
                self._rollback_or_fail_stop(transaction, error)
                raise
            finally:
                self.store.clear()

        if update_id:
            return self.update_cache.run(update_id, perform)
        return perform()

    def _op_update_impl(self, req):
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
        target_kl = req.get("target_kl", self.a.target_kl)
        assert_no_active_dropout(self.model)  # AUDIT FIX #5: no stochastic dropout in recompute
        torch.cuda.reset_peak_memory_stats()

        active = [(tid, chunks, max(-adv_clip, min(adv_clip, float(advantages.get(tid, 0.0)))))
                  for tid, chunks in self.store.items() if float(advantages.get(tid, 0.0)) != 0.0]
        adapter_before = self._adapter_vector()  # measurement fix: snapshot for adapter-delta
        epoch_stats = []
        grad_norm = 0.0
        n_chunks_last = 0
        for epoch in range(update_epochs):
            self.opt.zero_grad(set_to_none=True)
            max_logratio = math.log(ratio_max)
            flat = [(traj_id, chunk, adv) for traj_id, chunks, adv in active for chunk in chunks]
            old_parts = []; new_parts = []; pg_means = []; stat_meta = []
            for microbatch in self._microbatches(flat):
                batch_chunks = [item[1] for item in microbatch]
                new_batch = self._forward_new_logp_terms_batch(batch_chunks)
                micro_losses = []
                for (traj_id, chunk, adv), new_terms in zip(microbatch, new_batch):
                    # Faithful pi-RL recompute (shared with the post-step diagnostic): rebuild
                    # per-step means with the SAME corrected-drift equations and REUSE the stored
                    # per-step stds so epoch-0 ratio==1 exactly (on-policy correctness probe).
                    old_terms = self._old_logp_terms(chunk)
                    logratio = new_terms - old_terms
                    finite = torch.isfinite(logratio)
                    # PPO's clipped surrogate is already constant outside its clip band.
                    # Bound rare cross-architecture BF16 outliers for finite KL diagnostics
                    # instead of silently dropping their gradient terms from the batch.
                    valid_lr = torch.clamp(
                        torch.where(finite, logratio, torch.zeros_like(logratio)),
                        -max_logratio, max_logratio)
                    ratio = torch.exp(valid_lr)
                    adv_t = torch.as_tensor(adv, device=self.model.device)
                    pg = -torch.min(
                        ratio * adv_t, torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t)
                    kl = (ratio - 1.0) - valid_lr   # KL(new||old) approx, per scalar term
                    denominator = finite.sum().clamp_min(1).to(pg.dtype)
                    pg_mean = (pg * finite).sum() / denominator
                    loss = pg_mean + kl_coef * (kl * finite).sum() / denominator
                    micro_losses.append(loss)
                    old_parts.append(old_terms.detach().float())
                    new_parts.append(new_terms.detach().float())
                    pg_means.append(pg_mean.detach().float())
                    stat_meta.append((traj_id, int(chunk.get("denoise_index", -1)),
                                      float(adv), logratio.numel()))
                if micro_losses:
                    # One backward launch per microbatch instead of one per chunk.
                    torch.stack(micro_losses).sum().backward()
            import numpy as _np
            old_all = (torch.cat(old_parts).cpu().numpy() if old_parts
                       else _np.asarray([], dtype=_np.float32))
            new_all = (torch.cat(new_parts).cpu().numpy() if new_parts
                       else _np.asarray([], dtype=_np.float32))
            pg_values = (torch.stack(pg_means).cpu().numpy() if pg_means
                         else _np.asarray([], dtype=_np.float32))
            raw_logratios = new_all - old_all
            finite = _np.isfinite(raw_logratios)
            logratios = _np.clip(raw_logratios[finite], -max_logratio, max_logratio)
            ratios = _np.exp(logratios)
            n_nonfinite = int((~finite).sum())
            n_clamped = int((logratios != raw_logratios[finite]).sum())
            n_clipped = int((_np.abs(ratios - 1.0) > clip).sum())
            kl_sum = float(((ratios - 1.0) - logratios).sum())
            n_dropped = 0; n_chunks = 0; offset = 0; denoise_hist = {}; ratio_samples = []
            valid_chunk_flags = []
            old_logp_values = old_all[finite].tolist()
            new_logp_values = new_all[finite].tolist()
            for traj_id, denoise_index, adv, size in stat_meta:
                chunk_finite = finite[offset:offset + size]
                valid_count = int(chunk_finite.sum())
                n_dropped += int(valid_count == 0)
                n_chunks += int(valid_count > 0)
                valid_chunk_flags.append(valid_count > 0)
                denoise_hist[str(denoise_index)] = (
                    denoise_hist.get(str(denoise_index), 0) + valid_count)
                if len(ratio_samples) < 8 and valid_count:
                    positions = _np.flatnonzero(chunk_finite) + offset
                    for position in positions[:8 - len(ratio_samples)]:
                        bounded = float(_np.clip(raw_logratios[position],
                                                 -max_logratio, max_logratio))
                        ratio_samples.append({
                            "trajectory_id": traj_id, "denoise_index": denoise_index,
                            "advantage": adv, "old_logp": float(old_all[position]),
                            "new_logp": float(new_all[position]),
                            "ratio": math.exp(bounded),
                        })
                offset += size
            total_loss = (float(pg_values[_np.asarray(valid_chunk_flags)].sum())
                          if n_chunks else 0.0)
            gn = 0.0
            mean_kl_pre = kl_sum / max(len(ratios), 1)
            stop_for_kl = should_stop_for_kl(epoch, mean_kl_pre, target_kl)
            if n_chunks > 0 and not stop_for_kl:
                # Chunk-wise backward above accumulates a SUM. Divide gradients before
                # clipping/step so the optimizer sees the mean over valid active chunks.
                scale = mean_loss_scale(n_chunks)
                with torch.no_grad():
                    for p in self.trainable_params:
                        if p.grad is not None:
                            p.grad.mul_(scale)
                gn = float(torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=self.a.grad_clip))
                self.opt.step()
            # ESS-like weight diagnostic: (sum w)^2 / sum(w^2), normalized to [0,1]
            ess = None
            if ratios.size:
                w = ratios.astype(_np.float64, copy=False)
                ess = float((w.sum() ** 2) / ((w ** 2).sum() * len(w))) if (w ** 2).sum() > 0 else None
            epoch_stats.append({
                "epoch": epoch, "loss": total_loss / max(n_chunks, 1), "n_chunks": n_chunks,
                "n_dropped": n_dropped, "n_nonfinite": n_nonfinite,
                "n_clamped": n_clamped,
                "mean_ratio": (float(ratios.mean()) if ratios.size else 0.0),
                "mean_abs_logratio": (float(_np.abs(logratios).mean()) if logratios.size else 0.0),
                "clip_fraction": (n_clipped / len(ratios) if ratios.size else 0.0),
                "mean_kl": kl_sum / max(len(ratios), 1), "grad_norm": gn, "ess": ess,
                "n_ratio_terms": len(ratios),
                "old_logp_min": min(old_logp_values) if old_logp_values else None,
                "old_logp_max": max(old_logp_values) if old_logp_values else None,
                "old_logp_mean": (sum(old_logp_values) / len(old_logp_values)
                                   if old_logp_values else None),
                "new_logp_min": min(new_logp_values) if new_logp_values else None,
                "new_logp_max": max(new_logp_values) if new_logp_values else None,
                "new_logp_mean": (sum(new_logp_values) / len(new_logp_values)
                                   if new_logp_values else None),
                "denoise_ratio_term_histogram": denoise_hist,
                "ratio_samples": ratio_samples,
                "early_stop_target_kl": stop_for_kl,
            })
            grad_norm = gn; n_chunks_last = n_chunks
            if stop_for_kl:
                self.opt.zero_grad(set_to_none=True)
                break
        # measurement fix: adapter-parameter delta from this update (L2 + Linf over trainables)
        adapter_after = self._adapter_vector()
        with torch.no_grad():
            d = (adapter_after - adapter_before)
            adapter_delta_l2 = float(torch.linalg.vector_norm(d).cpu())
            adapter_delta_linf = float(d.abs().max().cpu()) if d.numel() else 0.0
            adapter_rel = adapter_delta_l2 / (float(torch.linalg.vector_norm(adapter_before).cpu()) + 1e-12)
        # measurement fix: recompute ratio/KL/clip/ESS AFTER the optimizer step (real
        # update-strength diagnostic, unlike the pre-step epoch-0 correctness probe).
        post = self._post_step_diagnostics(active, clip, ratio_max)
        first, last = epoch_stats[0], epoch_stats[-1]
        out = {"loss": last["loss"], "n_chunks": n_chunks_last, "n_dropped": last["n_dropped"],
                "grad_norm": grad_norm, "mean_ratio": last["mean_ratio"], "mean_kl": last["mean_kl"],
                "update_epochs": update_epochs, "epoch0_mean_ratio": first["mean_ratio"],
                "epochs_completed": len(epoch_stats),
                "early_stopped_target_kl": bool(epoch_stats[-1]["early_stop_target_kl"]),
                "epochLast_mean_ratio": last["mean_ratio"], "epochLast_clip_fraction": last["clip_fraction"],
                "epoch_stats": epoch_stats,
                "adapter_delta_l2": adapter_delta_l2, "adapter_delta_linf": adapter_delta_linf,
                "adapter_delta_rel": adapter_rel,
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        out.update(post)
        return out

    def op_save(self, req):
        path = Path(req["path"]); path.parent.mkdir(parents=True, exist_ok=True)
        sd = lora_state_dict(self.wrappers)
        # arms B/C also open non-LoRA params (action expert / vlm slice): save those too
        extra = {n: p.detach().cpu() for n, p in self.model.named_parameters()
                 if p.requires_grad and ".lora_" not in n}
        # Goal-3 reproducibility: persist optimizer + RNG + config + data/model provenance so a
        # resumed/loaded checkpoint is bit-reproducible and auditable (which data, which base rev).
        rng = {
            "torch": torch.get_rng_state(),
            "cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            "python": random.getstate(),
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
        metadata = build_checkpoint_metadata(
            adapter_skills=self.adapter_skills, targets=self.lora_target_modules,
            rank=self.a.rank, alpha=self.a.alpha, base_model=self.a.model,
            sampler=self.a.sampler, config=vars(self.a),
            source_commit=self.a.source_commit,
            source_manifest_sha256=self.a.source_manifest_sha256,
            update_index=int(req.get("update_index", 0)),
        )
        payload = {"lora": sd, "extra_trainable": extra, "config": vars(self.a),
                   "optimizer": self.opt.state_dict(), "rng": rng,
                   "provenance": provenance, "metadata": metadata,
                   "policy_version": self.policy_version,
                   "policy_hash": self.policy_hash,
                   "policy_hash_history": dict(self.policy_hash_history)}
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        return {"saved": str(path), "n_lora": len(sd), "n_extra": len(extra),
                "has_optimizer": True, "has_rng": True, "metadata": metadata,
                "data_hash": provenance["data_hash"], "model_rev": provenance["model_rev"],
                "policy_version": self.policy_version, "policy_hash": self.policy_hash}

    def op_load(self, req):
        if self.store:
            raise RuntimeError("cannot load checkpoint while rollout trajectories are active")
        blob = torch.load(req["path"], map_location=self.model.device)
        validate_checkpoint_metadata(
            blob.get("metadata"), adapter_skills=self.adapter_skills,
            targets=self.lora_target_modules, rank=self.a.rank, alpha=self.a.alpha)
        sd = blob["lora"]
        extra = blob.get("extra_trainable", {}) or {}
        named = dict(self.model.named_parameters())
        expected_extra = {n for n, p in self.model.named_parameters()
                          if p.requires_grad and ".lora_" not in n}
        if set(extra) != expected_extra:
            raise RuntimeError(
                f"extra_trainable layout mismatch: "
                f"missing={sorted(expected_extra-set(extra))[:3]} "
                f"unexpected={sorted(set(extra)-expected_extra)[:3]}")
        for name, tensor in extra.items():
            if tuple(named[name].shape) != tuple(tensor.shape):
                raise RuntimeError(
                    f"extra_trainable shape mismatch for {name}: "
                    f"expected={tuple(named[name].shape)} actual={tuple(tensor.shape)}")
        transaction = self._snapshot_mutable_state(include_optimizer=True)
        try:
            with torch.no_grad():
                load_lora_state_dict(self.wrappers, sd)
                for name, tensor in extra.items():
                    named[name].copy_(tensor.to(device=named[name].device, dtype=named[name].dtype))
            self.opt.load_state_dict(blob["optimizer"])
            rng = rng_state_for_restore(blob["rng"])
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
            if rng.get("python") is not None:
                random.setstate(rng["python"])
            if rng.get("numpy") is not None:
                import numpy as _np
                _np.random.set_state(rng["numpy"])
            self.policy_version = int(blob.get("policy_version", 0))
            self.policy_hash = self._compute_policy_hash()
            expected_hash = blob.get("policy_hash")
            if expected_hash is not None and expected_hash != self.policy_hash:
                raise RuntimeError("loaded checkpoint policy hash does not match its metadata")
            raw_history = blob.get("policy_hash_history") or {
                self.policy_version: self.policy_hash}
            history = {int(version): str(value) for version, value in raw_history.items()}
            known = history.get(self.policy_version)
            if known != self.policy_hash:
                raise RuntimeError("checkpoint policy history disagrees with current policy hash")
            self.policy_hash_history = history
            self._remember_policy_identity(self.policy_version, self.policy_hash)
        except Exception as error:
            self._rollback_or_fail_stop(transaction, error)
            raise
        return {"loaded": req["path"], "n_tensors": len(sd),
                "n_extra": len(extra), "metadata": blob["metadata"],
                "policy_version": self.policy_version, "policy_hash": self.policy_hash}

    def op_publish_adapter(self, req):
        """Atomically publish only mutable policy weights for the two rollout actors."""
        path = Path(req["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {n: p.detach().cpu() for n, p in self.model.named_parameters()
                 if p.requires_grad and ".lora_" not in n}
        payload = {"schema": "grpo-adapter-v1", "lora": lora_state_dict(self.wrappers),
                   "extra_trainable": extra, "policy_version": self.policy_version,
                   "policy_hash": self.policy_hash}
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        return {"published": str(path), "policy_version": self.policy_version,
                "policy_hash": self.policy_hash, "n_lora": len(payload["lora"]),
                "n_extra": len(extra)}

    def op_load_adapter(self, req):
        """Load an actor snapshot only between episodes; optimizer and RNG stay untouched."""
        if self.store:
            raise RuntimeError("cannot reload adapter while rollout trajectories are active")
        blob = torch.load(req["path"], map_location=self.model.device, weights_only=True)
        if blob.get("schema") != "grpo-adapter-v1":
            raise ValueError(f"unsupported adapter schema: {blob.get('schema')!r}")
        incoming_version = int(blob["policy_version"])
        incoming_hash = str(blob["policy_hash"])
        if incoming_version < self.policy_version:
            raise ValueError(
                f"refusing adapter downgrade v{self.policy_version} -> v{incoming_version}")
        known = self.policy_hash_history.get(incoming_version)
        if known is not None and known != incoming_hash:
            raise ValueError(
                f"policy version v{incoming_version} already names hash {known}, not {incoming_hash}")
        if incoming_version == self.policy_version and incoming_hash == self.policy_hash:
            return {"loaded_adapter": req["path"], "policy_version": self.policy_version,
                    "policy_hash": self.policy_hash, "idempotent": True}
        named = dict(self.model.named_parameters())
        extra = blob.get("extra_trainable") or {}
        expected_extra = {n for n, p in self.model.named_parameters()
                          if p.requires_grad and ".lora_" not in n}
        if set(extra) != expected_extra:
            raise RuntimeError("adapter extra_trainable layout mismatch")
        for name, tensor in extra.items():
            if name not in named or tuple(named[name].shape) != tuple(tensor.shape):
                raise RuntimeError(f"adapter extra_trainable mismatch: {name}")
        transaction = self._snapshot_mutable_state(include_optimizer=False)
        try:
            with torch.no_grad():
                load_lora_state_dict(self.wrappers, blob["lora"])
                for name, tensor in extra.items():
                    named[name].copy_(tensor.to(device=named[name].device, dtype=named[name].dtype))
            actual_hash = self._compute_policy_hash()
            if actual_hash != incoming_hash:
                raise RuntimeError("adapter hash mismatch after load")
            self.policy_version = incoming_version
            self.policy_hash = actual_hash
            self._remember_policy_identity(self.policy_version, self.policy_hash)
        except Exception as error:
            self._rollback_or_fail_stop(transaction, error)
            raise
        return {"loaded_adapter": req["path"], "policy_version": self.policy_version,
                "policy_hash": self.policy_hash}

    def op_reset(self, req):
        """Drop every buffered rollout chunk WITHOUT an optimizer step. The difficulty-band
        prefilter rolls out group members at the training eta>0 (populating self.store), but
        those scan rollouts must NOT enter the first real op_update batch. The client calls
        this after the prefilter (and after any diagnostic rollouts) to clear the store."""
        n_trajs = len(self.store)
        n_chunks = sum(len(v) for v in self.store.values())
        self.store.clear()
        return {"cleared_trajs": n_trajs, "cleared_chunks": n_chunks}

    def op_metrics(self, req):
        return {"n_trajs": len(self.store), "n_chunks": sum(len(v) for v in self.store.values()),
                "chunks_by_traj": {str(k): len(v) for k, v in self.store.items()},
                "free_gb": round(torch.cuda.mem_get_info()[0] / 1e9, 2),
                "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "policy_version": self.policy_version, "policy_hash": self.policy_hash,
                "policy_hash_history": dict(self.policy_hash_history),
                "fail_stopped": self.fatal_error is not None,
                "fatal_error": self.fatal_error}

    def op_export_store(self, req):
        """Persist selected collector trajectories without performing an update."""
        if export_rollout_store is None:
            raise RuntimeError("rollout_store integration is not mounted")
        ids = list(map(str, req["trajectory_ids"]))
        result = export_rollout_store(self.store, ids, Path(req["path"]),
                                      allowed_root=Path("/results"),
                                      actor_id=str(req.get("actor_id", "local")),
                                      group_id=str(req.get("group_id", "default")),
                                      policy_version=self.policy_version,
                                      policy_hash=self.policy_hash)
        if req.get("drop_after_export", False):
            for trajectory_id in ids:
                self.store.pop(trajectory_id, None)
            result["dropped_from_live_store"] = True
        return result

    def op_import_store(self, req):
        """Restore a collector payload so the existing op_update path can consume it."""
        if import_rollout_store is None:
            raise RuntimeError("rollout_store integration is not mounted")
        result = import_rollout_store(
            self.store, Path(req["path"]), expected_sha256=req["expected_sha256"],
            allowed_root=Path("/results"), learner_policy_version=self.policy_version,
            max_policy_lag=int(req.get("max_policy_lag", 1)))
        expected_hash = self.policy_hash_history.get(int(result["policy_version"]))
        if expected_hash is None or result["policy_hash"] != expected_hash:
            for trajectory_id in result["imported_trajectory_ids"]:
                self.store.pop(str(trajectory_id), None)
            raise ValueError(
                f"unrecognized policy identity v{result['policy_version']}/"
                f"{result['policy_hash']}; expected hash={expected_hash}")
        return result

    def op_discard_store(self, req):
        ids = list(map(str, req.get("trajectory_ids", [])))
        removed = []
        for trajectory_id in ids:
            if self.store.pop(trajectory_id, None) is not None:
                removed.append(trajectory_id)
        return {"discarded_trajectory_ids": removed, "optimizer_update_requested": False}

    def op_set_last_chunk_executed_steps(self, req):
        """Mask the unexecuted suffix of a terminal action chunk.

        Sampling records a full ``replan_steps`` action prefix before the simulator
        executes it.  An episode may succeed (or otherwise terminate) part-way through
        that prefix.  Without this correction, actions after the terminal environment
        step would still contribute policy-ratio terms even though they were never
        executed.  The operation only narrows the mask created at sampling time; it can
        never make previously masked actions trainable.
        """
        trajectory_id = str(req["trajectory_id"])
        executed_steps = int(req["executed_steps"])
        chunks = self.store.get(trajectory_id)
        if not chunks:
            raise KeyError(f"unknown rollout trajectory: {trajectory_id}")
        chunk = chunks[-1]
        mask = chunk["exec_mask_cpu"].to(torch.bool)
        if mask.ndim != 2:
            raise ValueError(f"expected [action_step, action_dim] mask, got {tuple(mask.shape)}")
        sampled_steps = int(mask.any(dim=1).sum().item())
        if executed_steps <= 0 or executed_steps > sampled_steps:
            raise ValueError(
                f"executed_steps must be in [1, {sampled_steps}], got {executed_steps}")
        narrowed = mask.clone()
        narrowed[executed_steps:] = False
        if bool((narrowed & ~mask).any()):  # defensive: this operation must be monotonic
            raise RuntimeError("terminal action mask attempted to enable new ratio terms")
        old_elements = chunk.get("old_logp_elements_cpu")
        if old_elements is None:
            raise RuntimeError(
                "terminal action-prefix masking requires elementwise rollout log-probabilities")
        chunk["exec_mask_cpu"] = narrowed
        chunk["old_logp"] = float(old_elements[narrowed].float().sum().item())
        chunk["executed_steps"] = executed_steps
        chunk["sampled_executed_steps"] = sampled_steps
        return {
            "trajectory_id": trajectory_id,
            "chunk_index": int(chunk.get("chunk_index", len(chunks) - 1)),
            "executed_steps": executed_steps,
            "sampled_executed_steps": sampled_steps,
            "masked_suffix_steps": sampled_steps - executed_steps,
        }

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
        if self.fatal_error is not None and req.get("op") not in ("metrics", "config"):
            raise RuntimeError(self.fatal_error)
        fn = {"sample": self.op_sample, "update": self.op_update, "save": self.op_save,
              "load": self.op_load, "metrics": self.op_metrics, "config": self.op_config,
              "publish_adapter": self.op_publish_adapter,
              "load_adapter": self.op_load_adapter,
              "reset": self.op_reset, "export_store": self.op_export_store,
              "import_store": self.op_import_store, "discard_store": self.op_discard_store,
              "set_last_chunk_executed_steps": self.op_set_last_chunk_executed_steps,
              "sft_update": self.op_sft_update, "sft_val": self.op_sft_val}.get(req.get("op"))
        return fn(req) if fn else {"error": f"unknown op {req.get('op')}"}

    def _serve_connection(self, conn):
        conn.settimeout(300.0)
        try:
            while True:
                ln = self._recv_all(conn, 4)
                if not ln:
                    break
                n = struct.unpack(">I", ln)[0]
                if n > 512 * 1024 * 1024:
                    raise ValueError(f"RPC request exceeds 512 MiB limit: {n}")
                payload = self._recv_all(conn, n)
                if not payload:
                    break
                with self.request_lock:
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

    def _serve_limited(self, conn, slots):
        try:
            self._serve_connection(conn)
        finally:
            slots.release()

    def serve(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.a.host, self.a.port))
            s.listen(128)
            print(f"GRPO trainer server on {self.a.host}:{self.a.port}...", flush=True)
            slots = threading.BoundedSemaphore(16)
            with ThreadPoolExecutor(max_workers=16, thread_name_prefix="grpo-rpc") as executor:
                while True:
                    conn, _ = s.accept()
                    if not slots.acquire(blocking=False):
                        conn.close()
                        continue
                    executor.submit(self._serve_limited, conn, slots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/checkpoint")
    ap.add_argument("--train-source-manifest", default="/train/source_manifest.json")
    ap.add_argument("--env-source-manifest", default="/rl_env/source_manifest.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=10088)
    ap.add_argument("--lr", type=float, default=2e-3)          # SGD LoRA (brief: 1e-3..5e-3)
    ap.add_argument("--optimizer", default="sgd")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--train-mode", default="adapter_only",
                    choices=("adapter_only", "adapter_plus_expert", "adapter_plus_expert_vlm"))
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--adapter-skills", default=",".join(SKILLS),
                    help="Comma-separated per-skill adapters to instantiate. Use 'grasp' for "
                         "a GRASP-only capacity experiment.")
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
    ap.add_argument("--target-kl", type=float, default=None,
                    help="Stop before a later-epoch optimizer step when mean KL exceeds this.")
    ap.add_argument("--microbatch-size", type=int, default=1,
                    help="Compatible rollout losses accumulated per backward launch.")
    ap.add_argument("--batch-model-forward", action="store_true",
                    help="Also collate model forwards (experimental; BF16 numerics differ).")
    ap.add_argument("--post-diag-chunks", type=int, default=128,
                    help="Evenly sampled chunks for post-update ratio diagnostics.")
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    ap.set_defaults(grad_checkpoint=True)
    a = ap.parse_args()
    validate_optimizer_config(a.update_epochs, a.ratio_max, a.clip, a.grad_clip, a.eta)
    if a.rank <= 0 or a.alpha <= 0:
        raise ValueError("rank and alpha must be positive")
    if a.lr <= 0.0:
        raise ValueError("lr must be positive")
    if a.microbatch_size <= 0 or a.post_diag_chunks <= 0:
        raise ValueError("microbatch-size and post-diag-chunks must be positive")
    train_source = verify_deployment_manifest("/train", a.train_source_manifest)
    verify_deployment_manifest("/rl_env", a.env_source_manifest)
    a.source_commit = train_source["commit"]
    a.source_manifest_sha256 = hashlib.sha256(
        Path(a.train_source_manifest).read_bytes()).hexdigest()
    GRPOTrainerServer(a).serve()


if __name__ == "__main__":
    main()
