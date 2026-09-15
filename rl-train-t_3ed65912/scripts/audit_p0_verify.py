"""P0 audit regression verification (GPU) for the CloseBlenderLid GRPO trainer.

Runs inside xiaomi-cu121 (GPU). Instantiates GRPOTrainerServer in-process (no socket)
and asserts the two P0 correctness fixes on the REAL checkpoint:

  FIX #1 -- adapter is ACTIVE in the deterministic (eta=0) eval path.
    * eta=0 sample with skill=None            == pretrained base (reference).
    * eta=0 sample with skill=grasp, B==0     == base            (zero-init LoRA is identity).
    * perturb lora_B[grasp] to nonzero, then
      eta=0 sample with skill=grasp           != base            (adapter now changes eval).
    * eta=0 sample with skill=None            == base            (skill isolation preserved).
    The pre-fix bug (set_active_skill(..., skill if eta>0 else None)) forced the eta=0
    eval to run the base policy, so before==after for every adapter-only arm.

  FIX #2 -- op_load restores the FULL trainable slice (LoRA + expert_proj + vlm_slice).
    * arm C (adapter_plus_expert_vlm): perturb LoRA + expert + vlm params, snapshot the
      eta=0 action, op_save, corrupt all trainable params, op_load, and require the eta=0
      action to match the snapshot within tolerance. The pre-fix op_load restored only
      LoRA, so expert/VLM reverted to pretrained and a post-hoc eval measured the wrong
      model.

Exit 0 iff both fixes verify.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, "/rl_env/src")
sys.path.insert(0, "/train/src")
from grpo_trainer_server import GRPOTrainerServer  # noqa: E402
from lora import select_trainable  # noqa: E402


def server_args(model, train_mode):
    return SimpleNamespace(
        model=model, host="127.0.0.1", port=0, lr=2e-3, optimizer="sgd",
        train_mode=train_mode, rank=8, alpha=32, lora_targets="qkv_proj",
        num_steps=5, eta=0.6, replan_steps=16, real_action_dim=12,
        clip=0.1, kl_coef=0.005, ratio_max=10.0, adv_clip=3.0, grad_clip=0.5,
        grad_checkpoint=True,
    )


def build_inputs(processor, robot_type, obs_history=4):
    from PIL import Image
    rng = np.random.default_rng(0)
    frames = [Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)) for _ in range(obs_history)]
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "Left camera: "}, {"type": "video", "video": frames},
            {"type": "text", "text": "\nRight camera: "}, {"type": "video", "video": frames},
            {"type": "text", "text": "\nWrist camera: "}, {"type": "video", "video": frames},
            {"type": "text", "text": "\n\nGenerate robot actions for the task:\nPick up the blender lid securely. /no_cot"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
    ]
    state = np.zeros((1, obs_history, 60), dtype=np.float32)
    state[0, :, :14] = rng.standard_normal((obs_history, 14)).astype(np.float32)
    inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True,
                                           return_tensors="pt", do_resize=False, state=state,
                                           robot_type=robot_type)
    d = dict(inputs)
    d["task_id"] = robot_type
    return d


def det_action(srv, inputs, skill, seed=12345):
    """One deterministic (eta=0) action chunk with the given active skill.

    The checkpoint's eta=0 Euler flow still draws its INITIAL latent from the global RNG,
    so we seed the global RNG before each call to make identical-input calls reproducible
    and thus isolate the effect of the active adapter (the quantity under test).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    resp = srv.op_sample({"inputs": {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                                     for k, v in inputs.items()},
                          "eta": 0.0, "skill": skill})
    return resp["actions"].float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/checkpoint")
    ap.add_argument("--robot-type", default="robocasa365")
    ap.add_argument("--out", default="/out/audit_p0_verify.json")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()
    from transformers import AutoProcessor
    rep = {"config": vars(args), "tests": {}}

    # ---------------- FIX #1: adapter active in eta=0 eval ----------------
    srv = GRPOTrainerServer(server_args(args.checkpoint, "adapter_only"))
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    inputs = build_inputs(processor, args.robot_type)

    base = det_action(srv, inputs, None)
    grasp_zeroinit = det_action(srv, inputs, "grasp")
    d_zeroinit = float(np.abs(base - grasp_zeroinit).max())

    # perturb ONLY the grasp adapter's B (breaks zero-init identity, simulates "trained")
    with torch.no_grad():
        for w in srv.wrappers:
            w.lora_B["grasp"].add_(torch.randn_like(w.lora_B["grasp"]) * 0.05)
    grasp_trained = det_action(srv, inputs, "grasp")
    none_after = det_action(srv, inputs, None)
    move_after = det_action(srv, inputs, "move_holding")  # untouched adapter, still zero-init
    d_trained = float(np.abs(base - grasp_trained).max())
    d_none = float(np.abs(base - none_after).max())
    d_move = float(np.abs(base - move_after).max())

    rep["tests"]["fix1_adapter_active_in_eval"] = {
        "zeroinit_grasp_equals_base": d_zeroinit <= args.tol, "d_zeroinit": round(d_zeroinit, 6),
        "trained_grasp_differs_from_base": d_trained > args.tol, "d_trained": round(d_trained, 6),
        "skill_none_still_base": d_none <= args.tol, "d_none": round(d_none, 6),
        "untouched_move_still_base": d_move <= args.tol, "d_move": round(d_move, 6),
    }
    f1 = rep["tests"]["fix1_adapter_active_in_eval"]
    fix1_pass = (f1["zeroinit_grasp_equals_base"] and f1["trained_grasp_differs_from_base"]
                 and f1["skill_none_still_base"] and f1["untouched_move_still_base"])
    del srv
    torch.cuda.empty_cache()

    # ---------------- FIX #2: full-slice checkpoint roundtrip (arm C) ----------------
    srv = GRPOTrainerServer(server_args(args.checkpoint, "adapter_plus_expert_vlm"))
    inputs = build_inputs(processor, args.robot_type)
    trainable_names = [n for n, p in srv.model.named_parameters() if p.requires_grad and ".lora_" not in n]
    rep["tests"]["fix2_checkpoint_roundtrip"] = {"n_extra_trainable": len(trainable_names)}

    # perturb LoRA + expert/vlm slice so the "trained" model differs from pretrained
    with torch.no_grad():
        for w in srv.wrappers:
            w.lora_B["grasp"].add_(torch.randn_like(w.lora_B["grasp"]) * 0.05)
        named = dict(srv.model.named_parameters())
        for n in trainable_names:
            named[n].add_(torch.randn_like(named[n]) * 0.02)
    ref = det_action(srv, inputs, "grasp")

    ckpt = "/out/_audit_roundtrip.pt"
    save_res = srv.op_save({"path": ckpt})

    # corrupt EVERY trainable param (lora + extra); a lora-only load would leave these wrong
    with torch.no_grad():
        for w in srv.wrappers:
            w.lora_B["grasp"].add_(torch.randn_like(w.lora_B["grasp"]) * 0.1)
        named = dict(srv.model.named_parameters())
        for n in trainable_names:
            named[n].add_(torch.randn_like(named[n]) * 0.05)
    corrupted = det_action(srv, inputs, "grasp")
    d_corrupt = float(np.abs(ref - corrupted).max())

    load_res = srv.op_load({"path": ckpt})
    restored = det_action(srv, inputs, "grasp")
    d_restore = float(np.abs(ref - restored).max())

    rep["tests"]["fix2_checkpoint_roundtrip"].update({
        "save": save_res, "load": load_res,
        "corruption_changed_output": d_corrupt > args.tol, "d_corrupt": round(d_corrupt, 6),
        "restore_matches_ref": d_restore <= args.tol, "d_restore": round(d_restore, 6),
        "n_extra_saved": save_res.get("n_extra"), "n_extra_loaded": load_res.get("n_extra"),
    })
    f2 = rep["tests"]["fix2_checkpoint_roundtrip"]
    fix2_pass = (f2["corruption_changed_output"] and f2["restore_matches_ref"]
                 and f2["n_extra_saved"] and f2["n_extra_saved"] == f2["n_extra_loaded"])

    rep["fix1_pass"] = bool(fix1_pass)
    rep["fix2_pass"] = bool(fix2_pass)
    # SCOPE: this script covers ONLY the two adapter/checkpoint P0 fixes (audit item set
    # "P0 adapter+checkpoint"). It does NOT certify the whole RL environment. The broader
    # addendum gates (deterministic x0, per-chunk RNG, real simulator restore, dropout,
    # multi-epoch ratio, checkpoint/resume) are verified by separate scripts/gates.
    rep["scope"] = "p0_adapter_checkpoint_subset_only"
    rep["p0_subset_pass"] = bool(fix1_pass and fix2_pass)
    rep["overall_pass"] = bool(fix1_pass and fix2_pass)  # legacy alias == p0_subset_pass
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0 if rep["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
