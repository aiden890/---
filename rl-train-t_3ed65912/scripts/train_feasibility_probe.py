"""GRPO training-step feasibility probe (GPU, xiaomi-cu121 image).

Decisive pre-build measurement for task t_3ed65912: can a REAL GRPO training
micro-step run on the shared RTX 3090 while the shared inference server
(xiaomi-server-t_460aea68, ~10.5 GB) stays resident?

A training step (unlike the env card's grad=False rollout) needs, per group:
  * one VLM forward (frozen, no_grad) to build conditioning,
  * `num_steps` DiT forwards WITH grad (the flow-SDE mean mu_k = x_k + v_theta*dt),
  * a GRPO clipped-ratio loss weighted by group-relative advantages,
  * backward through the DiT + projectors (VLM frozen), and an optimizer step.

This probe does exactly that on the real checkpoint with synthetic conditioning
(no simulator needed to measure memory/time), sweeping group_size and optimizer,
and reports peak CUDA memory + per-step wall time so we can pick a config that
fits alongside the shared server -- or report a hard blocker if nothing fits.

It imports the env card's flow_sde / flow_policy (single source of truth); it adds
only the training-step scaffolding this card owns.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# env card = single source of truth for the sampler + velocity-field adapter
_ENV_SRC = Path("/rl_env/src")
sys.path.insert(0, str(_ENV_SRC))
from flow_policy import build_velocity_field  # noqa: E402
from flow_sde import flow_sde_sample, transition_logprob, make_executed_mask  # noqa: E402

TRAINABLE_PREFIXES = (
    "dit.", "state_projector.", "action_projector.", "action_output_layer.",
    "t_embedder.", "t_projector.", "sink.",
)


def set_trainable(model):
    """Freeze the VLM backbone; train only the action-expert (DiT + projectors)."""
    trainable, frozen = [], 0
    for name, p in model.named_parameters():
        if any(name.startswith(pre) for pre in TRAINABLE_PREFIXES):
            p.requires_grad_(True)
            trainable.append(p)
        else:
            p.requires_grad_(False)
            frozen += p.numel()
    n_train = sum(p.numel() for p in trainable)
    return trainable, n_train, frozen


def build_dummy_conditioning(processor, robot_type, device, dtype, group, obs_history=4):
    """Build one group's worth of processor inputs from synthetic frames/state."""
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
    st = data.pop("state")
    am = data.pop("action_mask")
    # replicate to a group sharing identical conditioning (GRPO group == same start)
    st_g = st.repeat(group, 1, 1)
    am_g = am.repeat(group, 1, 1)
    vlm_g = {}
    for k, v in data.items():
        vlm_g[k] = v.repeat(group, *([1] * (v.dim() - 1))) if isinstance(v, torch.Tensor) and v.dim() >= 1 else v
    return st_g, am_g, vlm_g


def grpo_training_step(model, st_g, am_g, vlm_g, *, num_steps, eta, group,
                       replan_steps, real_action_dim, optimizer, clip=0.2,
                       grad_checkpoint=False):
    """One real GRPO step on synthetic conditioning; returns loss float."""
    # 1) sample a group of trajectories with the CURRENT policy, grad OFF (behaviour policy)
    with torch.no_grad():
        vfield, shape, dev, dt = build_velocity_field(model, st_g, am_g, **vlm_g)
        gen = torch.Generator(device="cuda").manual_seed(0)
        exec_mask = make_executed_mask(shape[1], shape[2], replan_steps, real_action_dim, device=dev)
        old = flow_sde_sample(vfield, shape, num_steps=num_steps, eta=eta, device=dev,
                              dtype=am_g.dtype, generator=gen, executed_mask=exec_mask)
    old_logp = old.executed_logprob().detach()                 # [G]
    xs = [x.detach() for x in old.xs]                          # frozen taken actions

    # 2) synthetic group-relative advantages (real reward comes from the sim in the loop)
    fake_returns = torch.randn(group, device=dev)
    adv = (fake_returns - fake_returns.mean()) / (fake_returns.std() + 1e-8)

    # 3) recompute log-prob under the CURRENT params WITH grad, through the DiT.
    #    Gradient checkpointing wraps each flow-step DiT forward so activations are
    #    recomputed in backward -- this cuts the dominant (group x num_steps x 36-layer)
    #    activation term, the single lever that lets group_size grow on the 3090.
    optimizer.zero_grad(set_to_none=True)
    vfield2, shape2, _, _ = build_velocity_field(model, st_g, am_g, **vlm_g)  # grad on
    dtc = 1.0 / num_steps
    means = []
    for k in range(num_steps):
        t_k = torch.full((group, 1, 1), k / num_steps, device=dev, dtype=am_g.dtype)
        if grad_checkpoint:
            v = torch.utils.checkpoint.checkpoint(vfield2, xs[k], t_k, use_reentrant=False)
        else:
            v = vfield2(xs[k], t_k)
        means.append(xs[k] + v * dtc)
    new_logp = transition_logprob(xs, means, eta=eta, num_steps=num_steps, executed_mask=exec_mask)  # [G]

    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
    loss = -torch.min(unclipped, clipped).mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/checkpoint")
    ap.add_argument("--robot-type", default="robocasa365")
    ap.add_argument("--num-steps", type=int, default=5)
    ap.add_argument("--eta", type=float, default=0.6)
    ap.add_argument("--replan-steps", type=int, default=16)
    ap.add_argument("--real-action-dim", type=int, default=12)
    ap.add_argument("--groups", default="2,4")
    ap.add_argument("--optims", default="adamw,sgd")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", default="/out/train_feasibility.json")
    args = ap.parse_args()

    from transformers import AutoModel, AutoProcessor
    device, dtype = "cuda", torch.bfloat16
    report = {"config": vars(args), "gpu": {}, "runs": []}
    report["gpu"]["name"] = torch.cuda.get_device_name(0)
    free0, total0 = torch.cuda.mem_get_info()
    report["gpu"]["free_before_load_gb"] = round(free0 / 1e9, 2)
    report["gpu"]["total_gb"] = round(total0 / 1e9, 2)

    model = AutoModel.from_pretrained(
        args.checkpoint, trust_remote_code=True, attn_implementation="flash_attention_2", dtype=dtype
    ).cuda().to(dtype)
    model.eval()  # frozen VLM in eval; DiT still trains (no dropout/bn issues here)
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)

    trainable, n_train, n_frozen = set_trainable(model)
    report["params"] = {"trainable": int(n_train), "frozen": int(n_frozen),
                        "trainable_gb_bf16": round(n_train * 2 / 1e9, 3)}
    torch.cuda.synchronize()
    report["gpu"]["used_after_load_gb"] = round((total0 - torch.cuda.mem_get_info()[0]) / 1e9, 2)

    for group in [int(g) for g in args.groups.split(",")]:
        for optim_name in [o.strip() for o in args.optims.split(",")]:
            run = {"group": group, "optimizer": optim_name}
            try:
                if optim_name == "adamw":
                    optimizer = torch.optim.AdamW(trainable, lr=1e-5)
                elif optim_name == "adamw8bit":
                    import bitsandbytes as bnb
                    optimizer = bnb.optim.AdamW8bit(trainable, lr=1e-5)
                elif optim_name == "sgd":
                    optimizer = torch.optim.SGD(trainable, lr=1e-4, momentum=0.0)
                else:
                    raise ValueError(optim_name)

                st_g, am_g, vlm_g = build_dummy_conditioning(processor, args.robot_type, device, dtype, group)
                torch.cuda.reset_peak_memory_stats()
                losses, times = [], []
                for s in range(args.steps):
                    t0 = time.time()
                    loss = grpo_training_step(
                        model, st_g, am_g, vlm_g, num_steps=args.num_steps, eta=args.eta,
                        group=group, replan_steps=args.replan_steps,
                        real_action_dim=args.real_action_dim, optimizer=optimizer,
                        grad_checkpoint=args.grad_checkpoint)
                    torch.cuda.synchronize()
                    times.append(time.time() - t0)
                    losses.append(loss)
                run["ok"] = True
                run["losses"] = losses
                run["step_time_s"] = round(float(np.mean(times[1:])) if len(times) > 1 else times[0], 3)
                run["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                run["free_after_gb"] = round(torch.cuda.mem_get_info()[0] / 1e9, 2)
                del optimizer, st_g, am_g, vlm_g
                torch.cuda.empty_cache()
            except RuntimeError as e:
                run["ok"] = False
                run["error"] = str(e)[:400]
                torch.cuda.empty_cache()
            report["runs"].append(run)
            print(json.dumps(run), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("=== REPORT ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
