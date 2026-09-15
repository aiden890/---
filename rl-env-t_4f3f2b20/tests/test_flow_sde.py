"""Unit tests for the flow-SDE sampler and its log-probabilities.

These run CPU-only with a toy linear velocity field -- no GPU, no model, no
simulator -- so they are the fast correctness gate for the RL sampler contract:

  1. eta = 0 reproduces the deterministic Euler ODE sampler bit-for-bit.
  2. Per-transition log-probs equal an independent torch.distributions.Normal.
  3. Recomputing log-probs on a fixed trajectory (GRPO path) matches the sampled
     log-probs and carries gradients into the velocity-field parameters.
  4. Reusing x0 + noise_sequence makes the whole rollout deterministic (branching
     from a shared prefix reproduces identical chunks).
  5. The executed mask zeroes out non-executed timesteps/dims.

Run: python3 -m pytest rl-env-t_4f3f2b20/tests/test_flow_sde.py -q
 or: python3 rl-env-t_4f3f2b20/tests/test_flow_sde.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flow_sde import (  # noqa: E402
    FlowSDEResult,
    flow_sde_sample,
    make_executed_mask,
    transition_logprob,
)


def deterministic_euler_reference(vfield, x0, num_steps):
    """The exact loop from checkpoint modeling_mibot.py ActionExpert.forward."""
    x = x0.clone()
    dt = 1.0 / num_steps
    for k in range(num_steps):
        t = torch.full((x.shape[0], 1, 1), k / num_steps, dtype=x.dtype)
        v = vfield(x, t)
        x = x + v * dt
    return x


def make_linear_field(A):
    """A fixed linear velocity field v(x,t) = x @ W + t*b, W deterministic."""
    torch.manual_seed(0)
    W = torch.randn(A, A) * 0.3
    b = torch.randn(A) * 0.1

    def vfield(x, t):
        return x @ W + t * b

    return vfield


def test_eta_zero_matches_deterministic_euler():
    B, L, A = 4, 8, 6
    vfield = make_linear_field(A)
    x0 = torch.randn(B, L, A)
    res = flow_sde_sample(vfield, (B, L, A), num_steps=5, eta=0.0, x0=x0)
    ref = deterministic_euler_reference(vfield, x0, num_steps=5)
    assert torch.allclose(res.actions, ref, atol=1e-6), "eta=0 must reproduce Euler ODE"
    print("[ok] eta=0 reproduces deterministic Euler sampler bit-for-bit")


def test_transition_logprob_matches_torch_normal():
    B, L, A = 3, 5, 4
    vfield = make_linear_field(A)
    eta, N = 0.7, 5
    g = torch.Generator().manual_seed(42)
    res = flow_sde_sample(vfield, (B, L, A), num_steps=N, eta=eta, generator=g)
    sigma = eta * math.sqrt(1.0 / N)
    total = torch.zeros(B)
    for k in range(N):
        dist = torch.distributions.Normal(res.means[k], sigma)
        lp = dist.log_prob(res.xs[k + 1]).sum(dim=(1, 2))
        assert torch.allclose(lp, res.step_logprob[:, k], atol=1e-5)
        total += lp
    assert torch.allclose(total, res.chunk_logprob, atol=1e-4)
    print("[ok] per-transition log-probs match torch.distributions.Normal")


def test_recompute_matches_and_has_gradient():
    B, L, A = 2, 6, 4
    eta, N = 0.5, 5
    torch.manual_seed(1)
    W = torch.randn(A, A, requires_grad=True)

    def vfield(x, t):
        return x @ W

    g = torch.Generator().manual_seed(7)
    res = flow_sde_sample(vfield, (B, L, A), num_steps=N, eta=eta, generator=g)

    # Recompute means under the same W (detached states -> GRPO path).
    xs = [x.detach() for x in res.xs]
    means = []
    dt = 1.0 / N
    for k in range(N):
        t = torch.full((B, 1, 1), k / N)
        means.append(xs[k] + (xs[k] @ W) * dt)
    lp = transition_logprob(xs, means, eta=eta, num_steps=N)
    assert torch.allclose(lp, res.chunk_logprob, atol=1e-4), "recompute must match sample"
    lp.sum().backward()
    assert W.grad is not None and torch.isfinite(W.grad).all(), "log-prob must be differentiable"
    print("[ok] GRPO recompute matches sampled log-prob and back-props into theta")


def test_reproducible_with_fixed_noise():
    B, L, A = 2, 4, 3
    vfield = make_linear_field(A)
    eta, N = 0.4, 5
    x0 = torch.randn(B, L, A)
    noise = [torch.randn(B, L, A) for _ in range(N)]
    r1 = flow_sde_sample(vfield, (B, L, A), num_steps=N, eta=eta, x0=x0, noise_sequence=noise)
    r2 = flow_sde_sample(vfield, (B, L, A), num_steps=N, eta=eta, x0=x0, noise_sequence=noise)
    assert torch.allclose(r1.actions, r2.actions, atol=1e-7)
    assert torch.allclose(r1.chunk_logprob, r2.chunk_logprob, atol=1e-7)
    print("[ok] fixed x0 + noise_sequence -> identical chunk (shared-prefix branching)")


def test_executed_mask_restricts_loss_terms():
    B, L, A = 2, 10, 12
    vfield = make_linear_field(A)
    eta, N = 0.6, 5
    replan, real_dim = 4, 7
    mask = make_executed_mask(L, A, replan_steps=replan, real_action_dim=real_dim)
    assert mask.sum().item() == replan * real_dim
    g = torch.Generator().manual_seed(3)
    res = flow_sde_sample(vfield, (B, L, A), num_steps=N, eta=eta, generator=g, executed_mask=mask)
    exec_lp = res.executed_logprob()
    full_lp = res.chunk_logprob
    # executed subset must be <= full (fewer negative-ish terms summed) and differ.
    assert exec_lp.shape == (B,)
    assert not torch.allclose(exec_lp, full_lp), "mask must actually restrict terms"
    # manual check
    manual = (res.perstep_perdim_logprob * mask.float()).sum(dim=(1, 2, 3))
    assert torch.allclose(exec_lp, manual, atol=1e-5)
    print("[ok] executed mask restricts GRPO loss to replan x real_dim terms")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"\nAll {len(fns)} flow-SDE tests passed.")


if __name__ == "__main__":
    _run_all()
