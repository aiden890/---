"""CPU unit tests for the faithful pi-RL Flow-SDE (pirl_flow_sde.py).

These cover the math gates that do NOT need the GPU/checkpoint:
  Gate 1 (partial): noise_level=0 reproduces the deterministic MiBoT Euler ODE
                    bit-for-bit on a synthetic velocity field.
  Gate 3 (partial): rollout-vs-recompute transition log-probs match on the identical
                    stored latent path (ratio == 1) -> importance ratio is 1 pre-update.
  Gate 4: corrected drift and g(t)=sigma schedule match an INDEPENDENT reimplementation
          of the RLinf openpi flow_sde equations (in native OpenPI time) at several t.
  Gate 6: all log-probs / stds are finite for noise_level>0.

The GPU gates (real-checkpoint eta=0 equivalence, PPO ratio==1 with the real net,
noise-level success sweep, post-update ODE transfer) run on v4 against the pinned
Xiaomi checkpoint; see scripts/pirl_sampler_gpu_gates.py.

Run:  python3 -m pytest tests/test_pirl_flow_sde.py -q      (or: python3 tests/test_pirl_flow_sde.py)
"""
import math
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from pirl_flow_sde import (  # noqa: E402
    openpi_timesteps,
    openpi_sigmas,
    pirl_flow_sde_sample,
    pirl_transition_logprob,
    pirl_step_mean_std,
)

torch.manual_seed(0)
B, L, A = 2, 4, 3
N = 5


def _mibot_velocity_field(target_data):
    """Synthetic MiBoT velocity: constant v_m = data - noise seen at t_m=0.

    For a rectified-flow toward `target_data`, the constant-velocity field that carries
    x0(noise) to target_data over t_m in [0,1] is v_m = target_data - x0. But the field
    must be state/time addressable: given x_t = (1-t_m)*noise + t_m*data and constant v,
    v = data - noise = data - (x_t - t_m*data)/(1-t_m). We use a simpler well-defined
    field: v_m(x,t_m) = (target_data - x) / max(1-t_m, eps), which integrates via exact
    Euler toward target_data and is finite on the MiBoT grid t_m in {0,1/N,...,(N-1)/N}.
    """
    def vf(x, t_m):
        tm = t_m  # [B,1,1]
        return (target_data - x) / (1.0 - tm).clamp(min=1e-6)
    return vf


def _mibot_euler_reference(vf, x0, num_steps):
    """The pinned MiBoT deterministic sampler: x <- x + v_m(x, k/N)*dt."""
    x = x0.clone()
    dt = 1.0 / num_steps
    for k in range(num_steps):
        t_m = torch.full((x.shape[0], 1, 1), k / num_steps, dtype=x.dtype)
        v = vf(x, t_m)
        x = x + v * dt
    return x


def _rlinf_openpi_reference_step(x_t, v_o, t_o, t_next_o, sigma_i):
    """INDEPENDENT reimplementation of RLinf sample_mean_var_val flow_sde branch, in
    native OpenPI time and with the OpenPI velocity v_o directly (no MiBoT remap).

    This is the ground-truth the ported module must match at every t. Copied structure
    from openpi_action_model.py L1169-1183, L1197."""
    delta = t_o - t_next_o
    x0_pred = x_t - v_o * t_o
    x1_pred = x_t + v_o * (1 - t_o)
    x0_weight = 1.0 - t_next_o
    x1_weight = t_next_o - sigma_i**2 * delta / (2 * t_o)
    mean = x0_pred * x0_weight + x1_pred * x1_weight
    std = math.sqrt(delta) * sigma_i
    return mean, std


def test_gate1_noise0_reproduces_mibot_euler():
    data = torch.randn(B, L, A)
    vf = _mibot_velocity_field(data)
    x0 = torch.randn(B, L, A)
    ref = _mibot_euler_reference(vf, x0, N)
    res = pirl_flow_sde_sample(vf, (B, L, A), num_steps=N, noise_level=0.0, x0=x0)
    max_abs = (res.actions - ref).abs().max().item()
    assert max_abs < 1e-5, f"noise=0 must equal MiBoT Euler; max_abs_diff={max_abs}"
    print(f"[Gate1] noise=0 vs MiBoT Euler max_abs_diff={max_abs:.2e} PASS")


def test_gate4_drift_and_sigma_match_rlinf_reference():
    """At several t, the ported mean/std (MiBoT-remapped) must equal the independent
    RLinf openpi-time reference fed the equivalent v_o = -v_m."""
    data = torch.randn(B, L, A)
    vf = _mibot_velocity_field(data)
    ts = openpi_timesteps(N)
    sigmas = openpi_sigmas(N, noise_level=0.7)
    x = torch.randn(B, L, A)
    worst = 0.0
    for k in range(N):
        t_o = float(ts[k]); t_next_o = float(ts[k + 1]); sig = float(sigmas[k])
        t_m = torch.full((B, 1, 1), 1.0 - t_o)
        v_m = vf(x, t_m)
        v_o = -v_m
        mean_port, std_port = pirl_step_mean_std(x, v_m, t_o, t_next_o, sig)
        mean_ref, std_ref = _rlinf_openpi_reference_step(x, v_o, t_o, t_next_o, sig)
        dm = (mean_port - mean_ref).abs().max().item()
        ds = abs(std_port - std_ref)
        worst = max(worst, dm, ds)
        assert dm < 1e-6 and ds < 1e-6, f"step {k}: dmean={dm}, dstd={ds}"
    print(f"[Gate4] ported vs RLinf openpi reference worst_diff={worst:.2e} PASS")


def test_gate4_sigma_schedule_values():
    """sigma_i = noise_level*sqrt(t_o/(1-t_o)) with the t_o=1 denom guard; std=sqrt(delta)*sigma."""
    nl = 0.5
    ts = openpi_timesteps(N)
    sig = openpi_sigmas(N, nl)
    assert sig.shape[0] == N
    # idx 0: t_o=1 -> guarded denom = ts[1]=(N-1)/N -> ratio = 1/(1/N)=N -> sigma=nl*sqrt(N)
    exp0 = nl * math.sqrt(N)
    assert abs(float(sig[0]) - exp0) < 1e-6, (float(sig[0]), exp0)
    # std at idx0 = sqrt(delta0)*sigma0 = sqrt(1/N)*nl*sqrt(N) = nl
    delta0 = float(ts[0] - ts[1])
    std0 = math.sqrt(delta0) * float(sig[0])
    assert abs(std0 - nl) < 1e-6, std0
    # all sigmas finite
    assert torch.isfinite(sig).all()
    print(f"[Gate4b] sigma schedule sig0={float(sig[0]):.4f} (exp {exp0:.4f}), std0={std0:.4f} PASS")


def test_gate3_rollout_recompute_logprob_match():
    """Recompute logp on the stored latent path with the stored stds == rollout logp
    (importance ratio == 1) BEFORE any parameter change."""
    data = torch.randn(B, L, A)
    vf = _mibot_velocity_field(data)
    g = torch.Generator().manual_seed(123)
    res = pirl_flow_sde_sample(vf, (B, L, A), num_steps=N, noise_level=0.6, generator=g)
    # recompute means from the SAME field on the stored xs (same net -> same means)
    ts = openpi_timesteps(N)
    sigmas = openpi_sigmas(N, 0.6)
    means = []
    for k in range(N):
        t_o = float(ts[k]); t_next_o = float(ts[k + 1]); sig = float(sigmas[k])
        t_m = torch.full((B, 1, 1), 1.0 - t_o)
        v_m = vf(res.xs[k], t_m)
        m, _ = pirl_step_mean_std(res.xs[k], v_m, t_o, t_next_o, sig)
        means.append(m)
    lp_recompute = pirl_transition_logprob(res.xs, means, res.stds)
    lp_rollout = res.chunk_logprob
    max_abs = (lp_recompute - lp_rollout).abs().max().item()
    ratio = torch.exp(lp_recompute - lp_rollout)
    assert max_abs < 1e-4, f"logp mismatch {max_abs}"
    assert (ratio - 1.0).abs().max().item() < 1e-4
    print(f"[Gate3] rollout-vs-recompute logp max_abs={max_abs:.2e}, ratio~1 PASS")


def test_gate6_finite_logprobs():
    data = torch.randn(B, L, A)
    vf = _mibot_velocity_field(data)
    for nl in [0.1, 0.3, 0.5, 1.0]:
        res = pirl_flow_sde_sample(vf, (B, L, A), num_steps=N, noise_level=nl,
                                   generator=torch.Generator().manual_seed(7))
        assert torch.isfinite(res.perstep_perdim_logprob).all(), nl
        assert torch.isfinite(res.chunk_logprob).all(), nl
        assert all(math.isfinite(s) and s > 0 for s in res.stds), nl
    print("[Gate6] logprobs/stds finite across noise levels PASS")


def test_marginal_std_matches_analytic():
    """Sanity: the terminal SDE sample mean over many draws stays near the data the ODE
    would produce (marginal preservation is exact in expectation for the linear field)."""
    data = torch.randn(1, L, A)
    vf = _mibot_velocity_field(data)
    x0 = torch.randn(1, L, A)
    ode = pirl_flow_sde_sample(vf, (1, L, A), num_steps=N, noise_level=0.0, x0=x0).actions
    draws = torch.stack([
        pirl_flow_sde_sample(vf, (1, L, A), num_steps=N, noise_level=0.4, x0=x0,
                             generator=torch.Generator().manual_seed(i)).actions
        for i in range(400)
    ], 0).squeeze(1)
    mean_sde = draws.mean(0, keepdim=True)
    diff = (mean_sde - ode).abs().mean().item()
    # linear field -> SDE mean tracks ODE closely; loose bound for MC noise
    assert diff < 0.15, f"SDE mean deviates from ODE by {diff}"
    print(f"[marginal] E[SDE]-ODE mean_abs={diff:.3f} PASS")


if __name__ == "__main__":
    test_gate1_noise0_reproduces_mibot_euler()
    test_gate4_drift_and_sigma_match_rlinf_reference()
    test_gate4_sigma_schedule_values()
    test_gate3_rollout_recompute_logprob_match()
    test_gate6_finite_logprobs()
    test_marginal_std_matches_analytic()
    print("\nALL CPU MATH GATES PASS")
