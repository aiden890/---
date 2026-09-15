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
from lora import (  # noqa: E402
    SKILLS, inject_per_skill_lora, set_active_skill, lora_parameters, select_trainable,
)
import vlm_scorer  # noqa: E402  (env card: single source of truth for the VQA prompt+math)


class GRPOTrainerServer:
    def __init__(self, a):
        from transformers import AutoModel
        self.a = a
        print("Loading model (GRPO per-skill-LoRA trainer)...", flush=True)
        self.model = AutoModel.from_pretrained(
            a.model, trust_remote_code=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16
        ).cuda().to(torch.bfloat16)
        self.model.eval()
        targets = tuple(t.strip() for t in a.lora_targets.split(","))
        self.wrappers = inject_per_skill_lora(self.model, skills=SKILLS, rank=a.rank,
                                              alpha=a.alpha, targets=targets)
        n_train, groups = select_trainable(self.model, self.wrappers, a.train_mode)
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"train_mode={a.train_mode} | per-skill LoRA rank={a.rank} into DiT {targets} | "
              f"trainable={n_train/1e6:.2f}M groups={groups} "
              f"({len(self.wrappers)} wrapped x {len(SKILLS)} skills)", flush=True)
        if a.optimizer == "sgd":
            self.opt = torch.optim.SGD(self.trainable_params, lr=a.lr, momentum=0.0)
        elif a.optimizer == "adamw":
            self.opt = torch.optim.AdamW(self.trainable_params, lr=a.lr)
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
        set_active_skill(self.wrappers, skill if eta > 0 else None)
        if eta == 0.0:
            with torch.no_grad():
                out = self.model(state=state, action_mask=action_mask, num_steps=self.a.num_steps, **vlm)
            return {"actions": out.actions.cpu(), "logprob": None}
        gen = None
        if req.get("seed") is not None:
            gen = torch.Generator(device="cuda").manual_seed(int(req["seed"]))
        with torch.no_grad():
            vfield, shape, dev, dt = build_velocity_field(self.model, state, action_mask, **vlm)
            exec_mask = make_executed_mask(shape[1], shape[2], self.a.replan_steps,
                                           self.a.real_action_dim, device=dev)
            res = flow_sde_sample(vfield, shape, num_steps=self.a.num_steps, eta=eta, device=dev,
                                  dtype=action_mask.dtype, generator=gen, executed_mask=exec_mask)
        old_logp = float(res.executed_logprob().float().cpu()[0])
        chunk = {
            "inputs_cpu": {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()},
            "xs_cpu": [x.detach().cpu() for x in res.xs],
            "exec_mask_cpu": exec_mask.cpu(),
            "old_logp": old_logp,
            "skill": skill,
        }
        self.store.setdefault(req["traj_id"], []).append(chunk)
        return {"actions": res.actions.cpu(), "logprob": old_logp}

    def op_update(self, req):
        advantages = req["advantages"]           # {traj_id: advantage float}
        clip = float(req.get("clip", self.a.clip))
        kl_coef = float(req.get("kl_coef", self.a.kl_coef))
        ratio_max = float(req.get("ratio_max", self.a.ratio_max))
        adv_clip = float(req.get("adv_clip", self.a.adv_clip))
        self.opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        total_loss = 0.0
        n_chunks = 0
        n_dropped = 0
        ratios = []
        kl_sum = 0.0
        for traj_id, chunks in self.store.items():
            adv = float(advantages.get(traj_id, 0.0))
            if adv == 0.0:
                continue
            adv = max(-adv_clip, min(adv_clip, adv))
            adv_t = torch.tensor(adv, device=self.model.device)
            for chunk in chunks:
                state, action_mask, vlm = self._split(dict(chunk["inputs_cpu"]))
                set_active_skill(self.wrappers, chunk["skill"])
                xs = [x.to(device=self.model.device, dtype=self.model.dtype) for x in chunk["xs_cpu"]]
                exec_mask = chunk["exec_mask_cpu"].to(self.model.device)
                old_logp = torch.tensor(chunk["old_logp"], device=self.model.device)
                vfield, shape, dev, dt = build_velocity_field(self.model, state, action_mask, **vlm)
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
                ratio = torch.exp(logratio)
                # ratio-explosion guard (brief): skip pathological samples
                if not torch.isfinite(ratio) or float(ratio) > ratio_max or float(ratio) < 1.0 / ratio_max:
                    n_dropped += 1
                    continue
                pg = -torch.min(ratio * adv_t, torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t)
                # KL(new||old) approx toward the behaviour policy (stabiliser, brief 2.5)
                kl = (torch.exp(logratio) - 1.0) - logratio
                loss = pg + kl_coef * kl
                loss.backward()
                total_loss += float(pg.detach().cpu())
                kl_sum += float(kl.detach().cpu())
                ratios.append(float(ratio.detach().cpu()))
                n_chunks += 1
        grad_norm = 0.0
        if n_chunks > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                self.trainable_params, max_norm=self.a.grad_clip))
            self.opt.step()
        self.store.clear()
        return {"loss": total_loss / max(n_chunks, 1), "n_chunks": n_chunks, "n_dropped": n_dropped,
                "grad_norm": grad_norm, "mean_ratio": (sum(ratios) / len(ratios) if ratios else 0.0),
                "mean_kl": kl_sum / max(n_chunks, 1),
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
        torch.save({"lora": sd, "extra_trainable": extra, "config": vars(self.a)}, path)
        return {"saved": str(path), "n_lora": len(sd), "n_extra": len(extra)}

    def op_load(self, req):
        blob = torch.load(req["path"], map_location=self.model.device)
        sd = blob["lora"]
        with torch.no_grad():
            for i, w in enumerate(self.wrappers):
                for s in w.skills:
                    w.lora_A[s].copy_(sd[f"w{i}.lora_A.{s}"].to(w.lora_A[s].dtype))
                    w.lora_B[s].copy_(sd[f"w{i}.lora_B.{s}"].to(w.lora_B[s].dtype))
        return {"loaded": req["path"], "n_tensors": len(sd)}

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

    def handle(self, req):
        fn = {"sample": self.op_sample, "update": self.op_update, "save": self.op_save,
              "load": self.op_load, "metrics": self.op_metrics,
              "vlm_score": self.op_vlm_score}.get(req.get("op"))
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
    ap.add_argument("--lora-targets", default="qkv_proj")
    ap.add_argument("--num-steps", type=int, default=5)
    ap.add_argument("--eta", type=float, default=0.6)
    ap.add_argument("--replan-steps", type=int, default=16)
    ap.add_argument("--real-action-dim", type=int, default=12)
    ap.add_argument("--clip", type=float, default=0.1)         # tighter than 0.2 (K=5 product)
    ap.add_argument("--kl-coef", type=float, default=0.005)
    ap.add_argument("--ratio-max", type=float, default=10.0)
    ap.add_argument("--adv-clip", type=float, default=3.0)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    ap.set_defaults(grad_checkpoint=True)
    a = ap.parse_args()
    GRPOTrainerServer(a).serve()


if __name__ == "__main__":
    main()
