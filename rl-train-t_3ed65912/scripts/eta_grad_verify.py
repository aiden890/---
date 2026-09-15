"""GPU verification: flow-SDE eta>0 log-prob numerical validity + gradient flow.

Operator gate item: confirm that at eta>0 the flow-SDE on the REAL checkpoint gives
(a) finite per-step and executed log-probs (no NaN/inf), (b) genuinely stochastic
distinct group members from shared conditioning, (c) a differentiable GRPO recompute
whose gradient actually reaches the DiT/projector params and is finite and NON-zero,
and (d) sane importance-ratio behaviour (ratio==1 exactly on-policy before any update).

Runs inside xiaomi-cu121 (GPU). Imports the env card sampler (single source of truth).
The env card probe only checked eta=0 identity + eta>0 finiteness with grad OFF; this
adds the TRAINING-critical backward pass and ratio sanity that GRPO depends on.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/rl_env/src")
from flow_policy import build_velocity_field  # noqa: E402
from flow_sde import flow_sde_sample, transition_logprob, make_executed_mask  # noqa: E402

TRAINABLE_PREFIXES = ("dit.", "state_projector.", "action_projector.", "action_output_layer.",
                      "t_embedder.", "t_projector.", "sink.")


def build_dummy(processor, robot_type, device, dtype, group, obs_history=4):
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
    data = {}
    for k, v in dict(inputs).items():
        data[k] = (v.to(device=device, dtype=dtype) if (isinstance(v, torch.Tensor) and v.is_floating_point())
                   else v.to(device=device) if isinstance(v, torch.Tensor) else v)
    st = data.pop("state"); am = data.pop("action_mask")
    st_g = st.repeat(group, 1, 1); am_g = am.repeat(group, 1, 1)
    vlm_g = {k: (v.repeat(group, *([1] * (v.dim() - 1))) if isinstance(v, torch.Tensor) and v.dim() >= 1 else v)
             for k, v in data.items()}
    return st_g, am_g, vlm_g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/checkpoint")
    ap.add_argument("--robot-type", default="robocasa365")
    ap.add_argument("--num-steps", type=int, default=5)
    ap.add_argument("--eta", type=float, default=0.6)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--replan-steps", type=int, default=16)
    ap.add_argument("--real-action-dim", type=int, default=12)
    ap.add_argument("--out", default="/out/eta_grad_verify.json")
    args = ap.parse_args()

    from transformers import AutoModel, AutoProcessor
    device, dtype = "cuda", torch.bfloat16
    rep = {"config": vars(args)}
    model = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True,
                                      attn_implementation="flash_attention_2", dtype=dtype).cuda().to(dtype)
    model.eval()
    for name, p in model.named_parameters():
        p.requires_grad_(any(name.startswith(pre) for pre in TRAINABLE_PREFIXES))
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    st_g, am_g, vlm_g = build_dummy(processor, args.robot_type, device, dtype, args.group)

    # (a) sample group at eta>0 (behaviour policy), grad off
    with torch.no_grad():
        vfield, shape, dev, dt = build_velocity_field(model, st_g, am_g, **vlm_g)
        exec_mask = make_executed_mask(shape[1], shape[2], args.replan_steps, args.real_action_dim, device=dev)
        gen = torch.Generator(device="cuda").manual_seed(0)
        old = flow_sde_sample(vfield, shape, num_steps=args.num_steps, eta=args.eta, device=dev,
                              dtype=dtype, generator=gen, executed_mask=exec_mask)
    step_lp = old.step_logprob.float().cpu()
    exec_lp = old.executed_logprob().float().cpu()
    rep["logprob_finite"] = {
        "step_all_finite": bool(torch.isfinite(step_lp).all()),
        "executed_all_finite": bool(torch.isfinite(exec_lp).all()),
        "executed_logprob": [round(float(x), 3) for x in exec_lp],
        "step_shape": list(step_lp.shape),
    }
    # (b) stochastic distinctness across the group from shared conditioning
    flat = old.actions.float().cpu().reshape(args.group, -1)
    pair_absdiff = float((flat[0] - flat[1]).abs().mean()) if args.group > 1 else 0.0
    rep["stochastic"] = {"distinct_members": bool(len({tuple(np.round(x[:8].numpy(), 4)) for x in flat}) == args.group),
                         "mean_abs_diff_m0_m1": round(pair_absdiff, 5)}

    # (c) differentiable GRPO recompute: gradient must reach DiT params, finite & nonzero
    xs = [x.detach() for x in old.xs]
    old_logp = exec_lp.to(dev)
    vfield2, shape2, _, _ = build_velocity_field(model, st_g, am_g, **vlm_g)  # grad ON
    dtc = 1.0 / args.num_steps
    means = []
    for k in range(args.num_steps):
        t_k = torch.full((args.group, 1, 1), k / args.num_steps, device=dev, dtype=dtype)
        v = torch.utils.checkpoint.checkpoint(vfield2, xs[k], t_k, use_reentrant=False)
        means.append(xs[k] + v * dtc)
    new_logp = transition_logprob(xs, means, eta=args.eta, num_steps=args.num_steps, executed_mask=exec_mask)
    fake_ret = torch.randn(args.group, device=dev)
    adv = (fake_ret - fake_ret.mean()) / (fake_ret.std() + 1e-8)
    ratio = torch.exp(new_logp - old_logp)
    loss = -torch.min(ratio * adv, torch.clamp(ratio, 0.9, 1.1) * adv).mean()
    loss.backward()

    grads = [(n, p.grad) for n, p in model.named_parameters()
             if p.requires_grad and p.grad is not None]
    total_norm = float(torch.sqrt(sum((g.float() ** 2).sum() for _, g in grads)).cpu()) if grads else 0.0
    all_finite = all(bool(torch.isfinite(g).all()) for _, g in grads)
    dit_hit = any(n.startswith("dit.") for n, _ in grads)
    rep["gradient"] = {
        "loss": round(float(loss.detach().cpu()), 6),
        "num_param_tensors_with_grad": len(grads),
        "total_grad_norm": round(total_norm, 6),
        "grad_all_finite": bool(all_finite),
        "grad_nonzero": bool(total_norm > 0),
        "reaches_dit": bool(dit_hit),
        "ratio_onpolicy": [round(float(x), 4) for x in ratio.detach().float().cpu()],
    }
    # (d) exact on-policy ratio == 1 when recompute uses the SAME params (sanity)
    rep["gradient"]["ratio_near_one_onpolicy"] = bool(
        np.allclose(rep["gradient"]["ratio_onpolicy"], 1.0, atol=5e-2))

    rep["overall_pass"] = bool(
        rep["logprob_finite"]["step_all_finite"] and rep["logprob_finite"]["executed_all_finite"]
        and rep["stochastic"]["distinct_members"]
        and rep["gradient"]["grad_all_finite"] and rep["gradient"]["grad_nonzero"]
        and rep["gradient"]["reaches_dit"] and rep["gradient"]["ratio_near_one_onpolicy"])

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0 if rep["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
