"""Flow-SDE sampler with per-transition log-probability for GRPO-style RL.

The Xiaomi-Robotics-1 action expert is a rectified-flow / flow-matching model.
At deployment it integrates a *deterministic* ODE (see checkpoint modeling_mibot.py
`ActionExpert.forward`, ~line 1864):

    x_0 ~ N(0, I)
    dt  = 1 / N
    for k in range(N):
        t_k     = k / N
        v_k     = v_theta(x_k, t_k)          # DiT velocity field
        x_{k+1} = x_k + v_k * dt             # explicit Euler
    return x_N                                # action chunk, shape [B, L, A]

This module turns that ODE into an Euler-Maruyama **SDE** so the policy becomes a
tractable stochastic distribution whose per-step transitions have closed-form
Gaussian log-probs. This is what GRPO needs: sample a group of chunks from the
*same* conditioning, score each transition's log-prob, and later recompute the
log-prob under the updated policy for the importance ratio.

SDE (isotropic exploration noise eta, drift = the model's own velocity):

    x_0 ~ N(0, I)
    dt  = 1 / N
    for k in range(N):
        t_k     = k / N
        mu_k    = x_k + v_theta(x_k, t_k) * dt
        sigma_k = eta * sqrt(dt)                       # scalar std
        eps_k   ~ N(0, I)
        x_{k+1} = mu_k + sigma_k * eps_k
        log p(x_{k+1} | x_k) = sum_dims log N(x_{k+1}; mu_k, sigma_k^2 I)

Properties (verified in tests/test_flow_sde.py):
  * eta = 0 reproduces the deterministic Euler sampler bit-for-bit.
  * The drift is unchanged, so the SDE keeps the model's learned dynamics and only
    injects exploration; eta controls the exploration/entropy trade-off.
  * Every transition is Gaussian, so log p is exact and differentiable w.r.t.
    theta (through mu_k = x_k + v_theta*dt) for the policy-gradient recompute.

GRPO masking: only the action timesteps/dimensions actually executed on the
simulator (the first `replan_steps` rows, the real `action_dim` columns) enter the
loss. `executed_logprob` applies that mask; `chunk_logprob` keeps the full sum for
diagnostics. The initial-noise term log p(x_0) is stored separately; it is identical
across a group with shared conditioning and cancels in the group-relative advantage,
but we keep it so the total trajectory log-prob is auditable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

# A velocity field: (x_k [B,L,A], t_k [B,1,1]) -> v_k [B,L,A].
VelocityField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_LOG_2PI = math.log(2.0 * math.pi)


def _gaussian_logprob(sample: torch.Tensor, mean: torch.Tensor, std: float) -> torch.Tensor:
    """Elementwise log N(sample; mean, std^2). Shape preserved (no reduction).

    Computed in float32 for numerical stability even when the states are bf16 (the
    model runs in bf16, but log-probs must be precise for the policy ratio)."""
    if std <= 0.0:
        raise ValueError("std must be > 0 for a well-defined Gaussian log-prob (eta > 0).")
    var = std * std
    s = sample.float()
    m = mean.float()
    return -0.5 * (((s - m) ** 2) / var + _LOG_2PI) - math.log(std)


@dataclass
class FlowSDEResult:
    """Output of a single flow-SDE integration for a batch/group of chunks."""

    actions: torch.Tensor                 # [B, L, A] final action chunk x_N
    x0: torch.Tensor                      # [B, L, A] initial noise x_0
    xs: list[torch.Tensor]                # N+1 states x_0..x_N (each [B, L, A])
    means: list[torch.Tensor]             # N per-step Euler means mu_k [B, L, A]
    step_logprob: torch.Tensor            # [B, N] summed-over-dims log p per SDE step
    perstep_perdim_logprob: torch.Tensor  # [B, N, L, A] elementwise transition log-probs
    x0_logprob: torch.Tensor              # [B] log p(x_0) under N(0,I)
    num_steps: int
    eta: float
    executed_mask: Optional[torch.Tensor] = None  # [L, A] bool, True = enters GRPO loss
    meta: dict = field(default_factory=dict)

    @property
    def chunk_logprob(self) -> torch.Tensor:
        """[B] full-chunk transition log-prob (sum over all steps & dims)."""
        return self.step_logprob.sum(dim=1)

    def executed_logprob(self) -> torch.Tensor:
        """[B] log-prob restricted to executed timesteps/dims (the GRPO term).

        Falls back to the full chunk log-prob when no mask was supplied.
        """
        if self.executed_mask is None:
            return self.chunk_logprob
        m = self.executed_mask.to(self.perstep_perdim_logprob.dtype)
        return (self.perstep_perdim_logprob * m).sum(dim=(1, 2, 3))


def make_executed_mask(
    chunk_len: int,
    action_dim: int,
    replan_steps: int,
    real_action_dim: Optional[int] = None,
    device=None,
) -> torch.Tensor:
    """Boolean [L, A] mask: first `replan_steps` rows and first `real_action_dim`
    columns are executed on the env and therefore enter the GRPO loss."""
    real_action_dim = real_action_dim if real_action_dim is not None else action_dim
    mask = torch.zeros((chunk_len, action_dim), dtype=torch.bool, device=device)
    rows = min(replan_steps, chunk_len)
    cols = min(real_action_dim, action_dim)
    mask[:rows, :cols] = True
    return mask


def flow_sde_sample(
    velocity_field: VelocityField,
    shape: tuple[int, int, int],
    num_steps: int,
    eta: float,
    *,
    device=None,
    dtype=torch.float32,
    generator: Optional[torch.Generator] = None,
    executed_mask: Optional[torch.Tensor] = None,
    x0: Optional[torch.Tensor] = None,
    noise_sequence: Optional[list[torch.Tensor]] = None,
) -> FlowSDEResult:
    """Integrate the flow-SDE and record every transition's log-prob.

    Args:
        velocity_field: the model's DiT velocity v_theta(x_k, t_k).
        shape: (B, L, A) = (group/batch, chunk length, action dim).
        num_steps: N Euler-Maruyama steps (matches the model's ODE num_steps).
        eta: exploration noise scale (sigma_k = eta * sqrt(dt)). eta=0 -> deterministic.
        x0: optional fixed initial noise (else sampled N(0,I)); enables branching from a
            shared prefix by reusing the same x0/noise across a group.
        noise_sequence: optional list of N pre-drawn eps_k for exact reproducibility.

    Returns:
        FlowSDEResult with actions, all intermediate states, means, and log-probs.
    """
    B, L, A = shape
    dt = 1.0 / num_steps

    if x0 is None:
        x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    else:
        x = x0.to(device=device, dtype=dtype).clone()

    # log p(x_0) under N(0, I), summed over dims -> [B] (float32)
    x0_logprob = (-0.5 * (x.float() ** 2 + _LOG_2PI)).sum(dim=(1, 2))

    xs = [x.clone()]
    means: list[torch.Tensor] = []
    perstep_perdim = torch.empty((B, num_steps, L, A), device=device, dtype=torch.float32)

    deterministic = eta == 0.0

    for k in range(num_steps):
        t_k = torch.full((B, 1, 1), k / num_steps, device=device, dtype=dtype)
        v_k = velocity_field(x, t_k)
        mu = x + v_k * dt
        means.append(mu.clone())

        if deterministic:
            x_next = mu
            # A degenerate (delta) transition: log-prob is not finite. We record 0 so
            # downstream code stays defined; RL rollouts must use eta > 0.
            perstep_perdim[:, k] = 0.0
        else:
            sigma = eta * math.sqrt(dt)
            if noise_sequence is not None:
                eps = noise_sequence[k].to(device=device, dtype=dtype)
            else:
                eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)
            x_next = mu + sigma * eps
            perstep_perdim[:, k] = _gaussian_logprob(x_next, mu, sigma)

        x = x_next
        xs.append(x.clone())

    step_logprob = perstep_perdim.sum(dim=(2, 3))  # [B, N]

    return FlowSDEResult(
        actions=xs[-1],
        x0=xs[0],
        xs=xs,
        means=means,
        step_logprob=step_logprob,
        perstep_perdim_logprob=perstep_perdim,
        x0_logprob=x0_logprob,
        num_steps=num_steps,
        eta=eta,
        executed_mask=executed_mask,
        meta={"dt": dt, "deterministic": deterministic},
    )


def transition_logprob(
    xs: list[torch.Tensor],
    means: list[torch.Tensor],
    eta: float,
    num_steps: int,
    executed_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Recompute per-step log p(x_{k+1}|x_k) for a *fixed* trajectory under a
    (possibly updated) policy that produced `means`.

    This is the GRPO recompute path: during optimisation the states xs are frozen
    (the actions actually taken), while `means[k] = x_k + v_theta_new(x_k,t_k)*dt`
    are differentiable functions of the current parameters theta. The returned
    log-prob therefore carries gradients into theta.

    Returns [B] executed log-prob if a mask is given, else full-chunk [B].
    """
    if eta <= 0.0:
        raise ValueError("eta must be > 0 to recompute a finite transition log-prob.")
    sigma = eta * math.sqrt(1.0 / num_steps)
    total = None
    per_terms = []
    for k in range(num_steps):
        lp = _gaussian_logprob(xs[k + 1], means[k], sigma)  # [B,L,A]
        per_terms.append(lp)
    stacked = torch.stack(per_terms, dim=1)  # [B,N,L,A]
    if executed_mask is not None:
        m = executed_mask.to(stacked.dtype)
        return (stacked * m).sum(dim=(1, 2, 3))
    return stacked.sum(dim=(1, 2, 3))
