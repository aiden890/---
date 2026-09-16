"""GPU validation gates for the FAITHFUL pi-RL Flow-SDE (pirl_flow_sde.py) on the REAL
pinned Xiaomi MiBoT checkpoint. Runs in-process inside xiaomi-cu121 (no simulator).

Covers the operator's pre-training validation gates that are checkable in-process:
  G1: noise=0 reproduces the checkpoint's deterministic ODE actions within tolerance.
  G2: rollout-time and recomputed old-policy transition log-probs match on the identical
      stored latent path (same params).
  G3: importance ratio == 1 within numerical tolerance for every retained sample before
      any optimizer step.
  G4: corrected drift + g(t)=sigma schedule are the pi-RL/RLinf equations (also unit-
      tested at several t on CPU in tests/test_pirl_flow_sde.py; here we print the
      realized schedule from the sampler meta for the record).
  G5(partial): ODE-vs-SDE action statistics across noise levels (per-noise mean/std of
      the executed action dims). The zero-shot SUCCESS sweep needs the simulator and is
      run separately via the rollout integration; this reports the action-distribution
      half so a noise level can be pre-screened.
  G6: all log-probs, ratios, KL, gradients finite; gradient reaches DiT; fail closed.
  G8: record framework commit / equations / checkpoint hash / config for reproducibility.

G5(success) and G7(post-update ODE transfer) require the sim rollout + a real update and
are gated AFTER this probe passes, in the rollout/trainer integration.

Reuses build_velocity_field + the dummy-input builder from the env card (single source of
truth). Imports the fixed_noise baseline too, ONLY to contrast schedules in the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from flow_policy import build_velocity_field  # noqa: E402
from pirl_flow_sde import (  # noqa: E402
    pirl_flow_sde_sample,
    pirl_transition_logprob,
    pirl_step_mean_std,
    openpi_timesteps,
    openpi_sigmas,
)
# make_executed_mask is shared in flow_sde.
from flow_sde import make_executed_mask  # noqa: E402

RLINF_COMMIT = "bde6c918642abf9a4776cb1d5fabcc5087dfe195"
TRAINABLE_PREFIXES = ("dit.", "state_projector.", "action_projector.", "action_output_layer.",
                      "t_embedder.", "t_projector.", "sink.")


def build_dummy(processor, robot_type, device, dtype, obs_history=4):
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
    return data


def _repeat(v, g):
    return v.repeat(g, *([1] * (v.dim() - 1))) if isinstance(v, torch.Tensor) and v.dim() >= 1 else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/checkpoint")
    ap.add_argument("--robot-type", default="robocasa365")
    ap.add_argument("--num-steps", type=int, default=5)
    ap.add_argument("--noise-level", type=float, default=0.5)
    ap.add_argument("--noise-sweep", default="0.1,0.3,0.5,0.7,1.0")
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--replan-steps", type=int, default=16)
    ap.add_argument("--real-action-dim", type=int, default=12)
    ap.add_argument("--eta0-tol", type=float, default=1e-2)   # bf16 tolerance
    ap.add_argument("--ratio-tol", type=float, default=5e-2)
    ap.add_argument("--out", default="/out/pirl_gpu_gates.json")
    args = ap.parse_args()

    from transformers import AutoModel, AutoProcessor
    device, dtype = "cuda", torch.bfloat16
    rep = {"config": vars(args), "framework_commit": RLINF_COMMIT,
           "sampler_file": "rl-env/src/pirl_flow_sde.py",
           "equations": "RLinf openpi flow_sde: sigma=nl*sqrt(t/(1-t)); "
                        "x1_weight=t_next - sigma^2*delta/(2t); std=sqrt(delta)*sigma; "
                        "MiBoT remap t_o=1-t_m, v_o=-v_m"}

    # checkpoint hash (config + weights index for provenance)
    try:
        cfg = Path(args.checkpoint) / "config.json"
        rep["checkpoint_config_sha256"] = hashlib.sha256(cfg.read_bytes()).hexdigest() if cfg.exists() else None
    except Exception as e:
        rep["checkpoint_config_sha256"] = f"err:{e}"

    model = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True,
                                      attn_implementation="flash_attention_2", dtype=dtype).cuda().to(dtype)
    model.eval()
    for name, p in model.named_parameters():
        p.requires_grad_(any(name.startswith(pre) for pre in TRAINABLE_PREFIXES))
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    rep["precision"] = "bfloat16"
    rep["denoise_steps"] = args.num_steps

    data = build_dummy(processor, args.robot_type, device, dtype)
    state = data.pop("state")
    action_mask = data.pop("action_mask")
    vlm = data

    # ---------- G1: noise=0 == checkpoint deterministic ODE (same x0) ----------
    torch.manual_seed(123)
    x0 = torch.randn_like(action_mask)
    real_randn_like = torch.randn_like
    torch.randn_like = lambda *a, **k: x0.clone()  # type: ignore
    try:
        with torch.no_grad():
            ckpt_out = model(state=state, action_mask=action_mask, num_steps=args.num_steps, **vlm)
        ckpt_actions = ckpt_out.actions.float().cpu()
    finally:
        torch.randn_like = real_randn_like  # type: ignore
    with torch.no_grad():
        vfield, shape, dev, dt = build_velocity_field(model, state, action_mask, **vlm)
        det = pirl_flow_sde_sample(vfield, shape, num_steps=args.num_steps, noise_level=0.0,
                                   device=dev, dtype=dtype, x0=x0)
    det_actions = det.actions.float().cpu()
    g1_max = (det_actions - ckpt_actions).abs().max().item()
    rep["G1_noise0_ode"] = {"max_abs_diff": g1_max, "tol": args.eta0_tol,
                            "pass": bool(g1_max < args.eta0_tol), "shape": list(det_actions.shape)}

    # ---------- group setup for stochastic gates ----------
    am_g = action_mask.repeat(args.group, 1, 1)
    st_g = state.repeat(args.group, 1, 1)
    vlm_g = {k: _repeat(v, args.group) for k, v in vlm.items()}

    with torch.no_grad():
        vfield_g, shape_g, dev_g, _ = build_velocity_field(model, st_g, am_g, **vlm_g)
        exec_mask = make_executed_mask(shape_g[1], shape_g[2], args.replan_steps,
                                       args.real_action_dim, device=dev_g)
        gen = torch.Generator(device="cuda").manual_seed(0)
        roll = pirl_flow_sde_sample(vfield_g, shape_g, num_steps=args.num_steps,
                                    noise_level=args.noise_level, device=dev_g, dtype=dtype,
                                    generator=gen, executed_mask=exec_mask)
    rep["G4_schedule"] = {"timesteps": roll.meta["ts"], "sigmas": roll.meta["sigmas"],
                          "stds": [round(s, 6) for s in roll.stds]}
    exec_lp_roll = roll.executed_logprob().float().cpu()
    step_lp = roll.step_logprob.float().cpu()

    # ---------- G2: recompute logp on stored path with SAME params == rollout logp ----------
    xs = [x.detach() for x in roll.xs]
    ts = openpi_timesteps(args.num_steps)
    with torch.no_grad():
        vfield_r, _, _, _ = build_velocity_field(model, st_g, am_g, **vlm_g)
        means_r = []
        sig = openpi_sigmas(args.num_steps, args.noise_level)
        for k in range(args.num_steps):
            t_o = float(ts[k]); t_next = float(ts[k + 1])
            t_m = torch.full((args.group, 1, 1), 1.0 - t_o, device=dev_g, dtype=dtype)
            v_m = vfield_r(xs[k], t_m)
            m, _ = pirl_step_mean_std(xs[k], v_m, t_o, t_next, float(sig[k]))
            means_r.append(m)
        exec_lp_recompute = pirl_transition_logprob(xs, means_r, roll.stds, executed_mask=exec_mask).float().cpu()
    g2_max = (exec_lp_recompute - exec_lp_roll).abs().max().item()
    rep["G2_rollout_recompute_logp"] = {"max_abs_diff": g2_max, "pass": bool(g2_max < 1e-2),
                                        "rollout": [round(float(x), 3) for x in exec_lp_roll],
                                        "recompute": [round(float(x), 3) for x in exec_lp_recompute]}

    # ---------- G3: importance ratio == 1 pre-update ----------
    ratio = torch.exp(exec_lp_recompute - exec_lp_roll)
    g3_ok = bool(np.allclose(ratio.numpy(), 1.0, atol=args.ratio_tol))
    rep["G3_ratio_onpolicy"] = {"ratios": [round(float(x), 5) for x in ratio], "tol": args.ratio_tol,
                                "pass": g3_ok}

    # ---------- G6: finiteness + differentiable update reaches DiT, finite/nonzero ----------
    finite_ok = bool(torch.isfinite(step_lp).all() and torch.isfinite(exec_lp_roll).all()
                     and torch.isfinite(ratio).all())
    old_logp = exec_lp_roll.to(dev_g)
    vfield_grad, _, _, _ = build_velocity_field(model, st_g, am_g, **vlm_g)  # grad ON
    means_g = []
    for k in range(args.num_steps):
        t_o = float(ts[k]); t_next = float(ts[k + 1])
        t_m = torch.full((args.group, 1, 1), 1.0 - t_o, device=dev_g, dtype=dtype)
        v_m = torch.utils.checkpoint.checkpoint(vfield_grad, xs[k], t_m, use_reentrant=False)
        m, _ = pirl_step_mean_std(xs[k], v_m, t_o, t_next, float(sig[k]))
        means_g.append(m)
    new_logp = pirl_transition_logprob(xs, means_g, roll.stds, executed_mask=exec_mask)
    fake_ret = torch.randn(args.group, device=dev_g)
    adv = (fake_ret - fake_ret.mean()) / (fake_ret.std() + 1e-8)
    r = torch.exp(new_logp - old_logp)
    kl = (old_logp - new_logp).mean()
    loss = -torch.min(r * adv, torch.clamp(r, 0.9, 1.1) * adv).mean()
    loss.backward()
    grads = [(n, p.grad) for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
    total_norm = float(torch.sqrt(sum((g.float() ** 2).sum() for _, g in grads)).cpu()) if grads else 0.0
    grad_finite = all(bool(torch.isfinite(g).all()) for _, g in grads)
    dit_hit = any(n.startswith("dit.") for n, _ in grads)
    rep["G6_finite_grad"] = {
        "logp_ratio_finite": finite_ok,
        "kl_finite": bool(torch.isfinite(kl)),
        "loss": round(float(loss.detach().cpu()), 6),
        "total_grad_norm": round(total_norm, 6),
        "grad_all_finite": grad_finite, "grad_nonzero": bool(total_norm > 0),
        "reaches_dit": dit_hit,
        "pass": bool(finite_ok and grad_finite and total_norm > 0 and dit_hit and torch.isfinite(kl)),
    }

    # ---------- G5(partial): ODE vs SDE action stats across noise levels ----------
    sweep = [float(x) for x in args.noise_sweep.split(",")]
    stats = {}
    real_dim = args.real_action_dim
    with torch.no_grad():
        # ODE reference
        vf1, sh1, d1, _ = build_velocity_field(model, state, action_mask, **vlm)
        ode = pirl_flow_sde_sample(vf1, sh1, num_steps=args.num_steps, noise_level=0.0,
                                   device=d1, dtype=dtype, x0=x0).actions.float().cpu()
        ode_exec = ode[..., :real_dim]
        for nl in sweep:
            accs = []
            for s in range(8):
                g = torch.Generator(device="cuda").manual_seed(1000 + s)
                a = pirl_flow_sde_sample(vf1, sh1, num_steps=args.num_steps, noise_level=nl,
                                         device=d1, dtype=dtype, generator=g, x0=x0.clone()).actions.float().cpu()
                accs.append(a[..., :real_dim])
            A = torch.stack(accs, 0)  # [8,1,L,real]
            dev_from_ode = (A.mean(0) - ode_exec).abs().mean().item()
            spread = A.std(0).mean().item()
            stats[f"{nl}"] = {"mean_abs_dev_from_ode": round(dev_from_ode, 5),
                              "sample_spread_std": round(spread, 5)}
    rep["G5_action_stats"] = {"note": "success sweep needs sim (separate rollout gate)",
                              "per_noise": stats}

    rep["overall_inprocess_pass"] = bool(
        rep["G1_noise0_ode"]["pass"] and rep["G2_rollout_recompute_logp"]["pass"]
        and rep["G3_ratio_onpolicy"]["pass"] and rep["G6_finite_grad"]["pass"])

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0 if rep["overall_inprocess_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
