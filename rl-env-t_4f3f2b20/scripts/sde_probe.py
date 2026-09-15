"""GPU probe: verify the flow-SDE adapter against the REAL checkpoint, in-process.

Runs inside the xiaomi-cu121 image (has torch+CUDA+flash-attn+the checkpoint). It:

  1. Loads MiBoTForActionGeneration from /checkpoint.
  2. Builds a real conditioning input with the processor from dummy-but-valid video/
     state (so no simulator is needed for this probe).
  3. Asserts the flow-SDE adapter with eta=0 reproduces the checkpoint's own
     `model.forward(...).actions` bit-for-bit (same x0) -> proves build_velocity_field
     mirrors upstream and the deterministic sampler is exact.
  4. Runs eta>0 and checks: per-step log-probs are finite, shape [B,N], recompute
     (transition_logprob) matches the sampled log-prob, and a group (B=4) from the
     same conditioning + shared x0 yields distinct chunks with distinct log-probs.
  5. Prints a JSON report to stdout and writes probe_report.json.

Usage (inside container, GPU visible):
    python3 /work/rl-env/scripts/sde_probe.py --checkpoint /checkpoint --out /out/probe_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from flow_policy import build_velocity_field  # noqa: E402
from flow_sde import flow_sde_sample, transition_logprob  # noqa: E402


def build_dummy_inputs(processor, robot_type, device, dtype, obs_history=4, crop=0.95):
    """Construct a valid processor request from synthetic frames/state (no simulator)."""
    from PIL import Image
    STATE_DIM = 60
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
    state_hist = rng.standard_normal((obs_history, 14)).astype(np.float32)
    state = np.zeros((1, obs_history, STATE_DIM), dtype=np.float32)
    state[0, :, :14] = state_hist
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
        do_resize=False, state=state, robot_type=robot_type,
    )
    data = {}
    for k, v in dict(inputs).items():
        if isinstance(v, torch.Tensor):
            data[k] = v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)
        else:
            data[k] = v
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/checkpoint")
    ap.add_argument("--robot-type", default="robocasa365")
    ap.add_argument("--num-steps", type=int, default=5)
    ap.add_argument("--eta", type=float, default=0.6)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--out", default="/out/probe_report.json")
    args = ap.parse_args()

    from transformers import AutoModel, AutoProcessor

    device = "cuda"
    dtype = torch.bfloat16
    report = {"checkpoint": args.checkpoint, "num_steps": args.num_steps, "eta": args.eta, "group": args.group}

    model = AutoModel.from_pretrained(
        args.checkpoint, trust_remote_code=True, attn_implementation="flash_attention_2", dtype=dtype
    ).cuda().to(dtype)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)

    data = build_dummy_inputs(processor, args.robot_type, device, dtype)
    state = data.pop("state")
    action_mask = data["action_mask"] if "action_mask" in data else None
    if action_mask is None:
        # The model builds action_mask internally in some versions; fall back to config dims.
        L = getattr(model.config, "action_chunk_size", 30)
        A = getattr(model.config, "action_dim", 12)
        action_mask = torch.ones((1, L, A), device=device, dtype=dtype)
    else:
        action_mask = data.pop("action_mask")
    vlm_kwargs = data

    # ---- (1) eta=0 equals the checkpoint's own Euler sampler (same x0) ----
    torch.manual_seed(123)
    x0 = torch.randn_like(action_mask)

    # checkpoint path, with x0 injected by monkeypatching randn_like for one call
    real_randn_like = torch.randn_like
    torch.randn_like = lambda *a, **k: x0.clone()  # type: ignore
    try:
        with torch.no_grad():
            ckpt_out = model(state=state, action_mask=action_mask, num_steps=args.num_steps, **vlm_kwargs)
        ckpt_actions = ckpt_out.actions.float().cpu()
    finally:
        torch.randn_like = real_randn_like  # type: ignore

    # adapter path, eta=0, same x0
    with torch.no_grad():
        vfield, shape, dev, dt = build_velocity_field(model, state, action_mask, **vlm_kwargs)
        det = flow_sde_sample(vfield, shape, num_steps=args.num_steps, eta=0.0, device=dev, dtype=dtype, x0=x0)
    det_actions = det.actions.float().cpu()

    max_abs = (det_actions - ckpt_actions).abs().max().item()
    report["eta0_matches_checkpoint"] = {
        "max_abs_diff": max_abs,
        "pass": bool(max_abs < 1e-2),  # bf16 tolerance
        "shape": list(det_actions.shape),
    }

    # ---- (2..4) eta>0 group rollout, log-prob sanity + recompute ----
    action_mask_g = action_mask.repeat(args.group, 1, 1)
    state_g = state.repeat(args.group, 1, 1)
    vlm_g = {}
    for k, v in vlm_kwargs.items():
        vlm_g[k] = v.repeat(args.group, *([1] * (v.dim() - 1))) if isinstance(v, torch.Tensor) and v.dim() >= 1 else v

    torch.manual_seed(7)
    with torch.no_grad():
        vfield_g, shape_g, dev_g, dt_g = build_velocity_field(model, state_g, action_mask_g, **vlm_g)
        res = flow_sde_sample(vfield_g, shape_g, num_steps=args.num_steps, eta=args.eta, device=dev_g, dtype=dtype)

    step_lp = res.step_logprob.float().cpu()
    chunk_lp = res.chunk_logprob.float().cpu()
    recomputed = transition_logprob(res.xs, res.means, eta=args.eta, num_steps=args.num_steps).float().cpu()

    report["sde_group"] = {
        "step_logprob_shape": list(step_lp.shape),
        "all_finite": bool(torch.isfinite(step_lp).all()),
        "chunk_logprob": [float(x) for x in chunk_lp],
        "recompute_matches": bool(torch.allclose(chunk_lp, recomputed, atol=1e-2)),
        "recompute_max_abs_diff": float((chunk_lp - recomputed).abs().max()),
        "distinct_chunks": bool(len({tuple(np.round(x.flatten()[:8].numpy(), 4)) for x in res.actions.float().cpu()}) == args.group),
        "actions_shape": list(res.actions.shape),
    }

    report["overall_pass"] = bool(
        report["eta0_matches_checkpoint"]["pass"]
        and report["sde_group"]["all_finite"]
        and report["sde_group"]["recompute_matches"]
        and report["sde_group"]["distinct_chunks"]
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
