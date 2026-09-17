"""Faithful pi-RL marginal-preserving Flow-SDE for the Xiaomi MiBoT action expert.

Provenance (operator requirement: do NOT reimplement the RL framework from memory;
use RLinf's OpenPI Flow-SDE as the source of truth for the ODE->SDE equations,
corrected drift, time-dependent noise schedule, and transition log-probability):

    Framework : RLinf/RLinf  commit bde6c918642abf9a4776cb1d5fabcc5087dfe195 (2026-09-16)
    File      : rlinf/models/embodiment/openpi/openpi_action_model.py
    Functions : OpenPIActionModel.sample_mean_var_val()  (flow_sde branch, ~L1176-1183)
                OpenPIActionModel._get_timesteps()        (~L1120)
                OpenPIActionModel.get_logprob_norm()      (~L1287)
                OpenPIActionModel.get_log_prob_value()    (recompute path, ~L1305)
    Paper     : pi-RL, arXiv:2510.25889 (Flow-SDE: ODE->SDE with equivalent marginals)

This module implements the SAME equations as RLinf's flow_sde branch, adapted ONLY by
the Xiaomi-specific velocity-field adapter (the MiBoT time/sign convention differs from
OpenPI's). Everything load-bearing -- the timestep grid, the corrected drift weights,
the tau-dependent sigma schedule sigma=noise_level*sqrt(t/(1-t)), the per-step std
sqrt(delta)*sigma, and the Gaussian transition log-prob -- is ported verbatim from the
RLinf source, NOT invented here.

CONVENTION MAP (this is the only Xiaomi-specific piece)
-------------------------------------------------------
OpenPI: time t_o runs 1 (pure noise) -> 0 (clean action); x_{t} = t_o*noise + (1-t_o)*data;
        the model velocity v_o points noise-ward (dx/dt_o = noise - data); denoising
        DECREASES t_o.
MiBoT : time t_m runs 0 (pure noise) -> 1 (clean action); x_{t} = (1-t_m)*noise + t_m*data;
        the model velocity v_m = dit_forward points data-ward (dx/dt_m = data - noise);
        integration INCREASES t_m via Euler x <- x + v_m*dt.

Relation:  t_o = 1 - t_m   and   v_o(x, t_o) = -v_m(x, 1 - t_o) = -v_m(x, t_m).

We keep every OpenPI coefficient in OpenPI time t_o (so the denominators match the source
and never divide by zero on the retained steps), and obtain v_o by calling the MiBoT
velocity field at t_m = 1 - t_o and negating. Verified consequences (see tests):
  * noise_level = 0 reproduces the pinned MiBoT deterministic Euler ODE bit-for-bit
    (the flow_ode weights reduce to x_next = x + v_m*dt on the identical t_m grid).
  * every SDE transition is an exact diagonal Gaussian, so the importance ratio the
    PPO/GRPO update needs is exact and tractable, and rollout-vs-recompute log-probs on
    the stored latent path are identical (ratio == 1) before any optimizer step.

This is DISTINCT from `flow_sde.py`, which is the explicitly-named fixed-noise baseline
(constant sigma = eta*sqrt(dt), UNCORRECTED drift mu = x + v*dt). That module must never
be labelled pi-RL, Flow-SDE, or marginal-preserving. Only THIS module may.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

# MiBoT velocity field: (x [B,L,A], t_m [B,1,1]) -> v_m [B,L,A].
VelocityField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_LOG_2PI = math.log(2.0 * math.pi)


def openpi_timesteps(num_steps: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """RLinf `_get_timesteps`: t_o grid = [1, (N-1)/N, ..., 1/N, 0], length N+1.

    Ported verbatim from openpi_action_model._get_timesteps (RLinf bde6c91)."""
    ts = torch.linspace(1.0, 1.0 / num_steps, num_steps, device=device, dtype=dtype)
    ts = torch.cat([ts, torch.zeros((1,), device=device, dtype=dtype)])
    return ts


def openpi_sigmas(num_steps: int, noise_level: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """RLinf flow_sde sigma schedule (length N, one per integration step).

    Verbatim from sample_mean_var_val flow_sde branch:
        denom = where(timesteps==1, timesteps[1], timesteps)   # guard t_o=1
        sigma_ratio = timesteps / (1 - denom)
        sigmas = noise_level * sqrt(sigma_ratio)[:-1]
    """
    ts = openpi_timesteps(num_steps, device=device, dtype=dtype)
    denom = torch.where(ts == 1.0, ts[1], ts)
    sigma_ratio = ts / (1.0 - denom)
    sigmas = noise_level * torch.sqrt(sigma_ratio)
    return sigmas[:-1]  # drop t_o = 0


def pirl_step_mean_std(
    x_t: torch.Tensor,
    v_m: torch.Tensor,
    t_o: float,
    t_next_o: float,
    sigma_i: float,
) -> tuple[torch.Tensor, float]:
    """One faithful pi-RL Flow-SDE transition, in OpenPI coefficients / MiBoT velocity.

    Args:
        x_t: current latent [B,L,A].
        v_m: MiBoT velocity at this step, v_m(x_t, t_m=1-t_o) [B,L,A].
        t_o: current OpenPI time timesteps[idx].
        t_next_o: next OpenPI time timesteps[idx+1] (= t_o - delta).
        sigma_i: openpi_sigmas[idx].

    Returns (mean [B,L,A], std scalar). Verbatim OpenPI flow_sde branch:
        v_o = -v_m
        x0_pred (data)  = x_t - v_o*t_o
        x1_pred (noise) = x_t + v_o*(1-t_o)
        x0_weight = 1 - t_next_o
        x1_weight = t_next_o - sigma_i^2 * delta / (2*t_o)     # corrected drift
        mean = x0_pred*x0_weight + x1_pred*x1_weight
        std  = sqrt(delta) * sigma_i
    """
    delta = t_o - t_next_o
    v_o = -v_m
    x0_pred = x_t - v_o * t_o           # data prediction
    x1_pred = x_t + v_o * (1.0 - t_o)   # noise prediction
    x0_weight = 1.0 - t_next_o
    x1_weight = t_next_o - (sigma_i * sigma_i) * delta / (2.0 * t_o)
    mean = x0_pred * x0_weight + x1_pred * x1_weight
    std = math.sqrt(delta) * sigma_i
    return mean, std


def _gaussian_logprob(sample: torch.Tensor, mean: torch.Tensor, std: float) -> torch.Tensor:
    """Elementwise log N(sample; mean, std^2), float32, no reduction.

    Mirrors RLinf get_logprob_norm (safe_get_logprob=False path). std==0 (the
    deterministic ODE limit) yields 0 so downstream stays defined; RL rollouts use std>0.
    """
    s = sample.float()
    m = mean.float()
    if std == 0.0:
        return torch.zeros_like(s)
    var = std * std
    return -0.5 * (((s - m) ** 2) / var + _LOG_2PI) - math.log(std)


@dataclass
class PiRLFlowSDEResult:
    actions: torch.Tensor                 # [B,L,A] final clean action (t_o -> 0)
    x0: torch.Tensor                      # [B,L,A] initial noise (t_o = 1)
    xs: list[torch.Tensor]                # N+1 latents x_0..x_N (the stored latent path)
    means: list[torch.Tensor]            # N per-step means
    stds: list[float]                     # N per-step scalar stds (time-dependent)
    perstep_perdim_logprob: torch.Tensor  # [B,N,L,A]
    step_logprob: torch.Tensor            # [B,N]
    x0_logprob: torch.Tensor              # [B] log N(x0;0,I)
    num_steps: int
    noise_level: float
    executed_mask: Optional[torch.Tensor] = None
    meta: dict = field(default_factory=dict)

    @property
    def chunk_logprob(self) -> torch.Tensor:
        return self.step_logprob.sum(dim=1)

    def executed_logprob(self) -> torch.Tensor:
        if self.executed_mask is None:
            return self.chunk_logprob
        m = self.executed_mask.to(self.perstep_perdim_logprob.dtype)
        return (self.perstep_perdim_logprob * m).sum(dim=(1, 2, 3))


def pirl_flow_sde_sample(
    velocity_field: VelocityField,
    shape: tuple[int, int, int],
    num_steps: int,
    noise_level: float,
    *,
    device=None,
    dtype=torch.float32,
    generator: Optional[torch.Generator] = None,
    executed_mask: Optional[torch.Tensor] = None,
    x0: Optional[torch.Tensor] = None,
    noise_sequence: Optional[list[torch.Tensor]] = None,
    denoise_index: Optional[int] = None,
) -> PiRLFlowSDEResult:
    """Integrate the faithful pi-RL Flow-SDE, recording every transition's log-prob.

    noise_level == 0 -> deterministic; equals the checkpoint's MiBoT Euler ODE.
    noise_level > 0  -> marginal-preserving stochastic policy with exact Gaussian
                        transition log-probs (corrected drift + tau-dependent sigma).
    """
    B, L, A = shape
    if denoise_index is not None and not 0 <= int(denoise_index) < num_steps:
        raise ValueError(f"denoise_index must be in [0, {num_steps}): {denoise_index}")
    ts = openpi_timesteps(num_steps, device=device, dtype=torch.float32)
    sigmas = openpi_sigmas(num_steps, noise_level, device=device, dtype=torch.float32)

    if x0 is None:
        x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    else:
        x = x0.to(device=device, dtype=dtype).clone()

    x0_logprob = (-0.5 * (x.float() ** 2 + _LOG_2PI)).sum(dim=(1, 2))

    xs = [x.clone()]
    means: list[torch.Tensor] = []
    stds: list[float] = []
    perstep = torch.empty((B, num_steps, L, A), device=device, dtype=torch.float32)

    for k in range(num_steps):
        t_o = float(ts[k].item())
        t_next_o = float(ts[k + 1].item())
        # MiBoT velocity is evaluated at t_m = 1 - t_o.
        t_m = torch.full((B, 1, 1), 1.0 - t_o, device=device, dtype=dtype)
        v_m = velocity_field(x, t_m)
        sigma_i = float(sigmas[k].item())

        stochastic_step = noise_level != 0.0 and (
            denoise_index is None or k == int(denoise_index)
        )
        if not stochastic_step:
            # Degenerate ODE limit. At sigma=0 the corrected-drift step reduces EXACTLY to
            # the checkpoint's Euler update x_{k+1} = x_k + v_m*delta (proof: with sigma=0,
            # x0_weight=1-t_next, x1_weight=t_next -> mean = x + v_m*(t_o-t_next)). We compute
            # it in that direct form (same op order/dtype as the checkpoint) so eta=0 is
            # bit-for-bit the pinned MiBoT ODE, rather than incurring bf16 error from the
            # x0_pred/x1_pred recombination that only the sigma>0 drift correction needs.
            delta = t_o - t_next_o
            mean = x + v_m * delta
            std = 0.0
        else:
            mean, std = pirl_step_mean_std(x, v_m, t_o, t_next_o, sigma_i)
        means.append(mean.clone())
        stds.append(std)

        if std == 0.0:
            x_next = mean
            perstep[:, k] = 0.0
        else:
            if noise_sequence is not None:
                eps = noise_sequence[k].to(device=device, dtype=dtype)
            else:
                eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)
            x_next = mean + std * eps
            perstep[:, k] = _gaussian_logprob(x_next, mean, std)

        x = x_next
        xs.append(x.clone())

    step_logprob = perstep.sum(dim=(2, 3))
    return PiRLFlowSDEResult(
        actions=xs[-1],
        x0=xs[0],
        xs=xs,
        means=means,
        stds=stds,
        perstep_perdim_logprob=perstep,
        step_logprob=step_logprob,
        x0_logprob=x0_logprob,
        num_steps=num_steps,
        noise_level=noise_level,
        executed_mask=executed_mask,
        meta={"ts": ts.tolist(), "sigmas": sigmas.tolist(),
              "framework": "RLinf@bde6c918", "sampler": "pirl_flow_sde",
              "denoise_index": denoise_index},
    )


def pirl_transition_logprob_elements(
    xs: list[torch.Tensor],
    means: list[torch.Tensor],
    stds: list[float],
    denoise_index: int,
) -> torch.Tensor:
    """Elementwise log-prob for RLinf's non-joint Flow-SDE transition.

    RLinf configures Flow-SDE with ``joint_logprob=False``: exactly one denoising
    transition is stochastic and PPO keeps its [B,L,A] ratios independent. Summing
    them before exponentiating creates a dimension-dependent joint ratio instead.
    """
    idx = int(denoise_index)
    if not 0 <= idx < len(means):
        raise ValueError(f"denoise_index must be in [0, {len(means)}): {idx}")
    if float(stds[idx]) <= 0.0:
        raise ValueError("selected Flow-SDE transition must have positive std")
    return _gaussian_logprob(xs[idx + 1], means[idx], stds[idx])


def pirl_transition_logprob(
    xs: list[torch.Tensor],
    means: list[torch.Tensor],
    stds: list[float],
    executed_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Recompute per-step log p(x_{k+1}|x_k) for a FIXED stored latent path under a
    (possibly updated) policy that produced `means`, using the stored per-step `stds`.

    This is the PPO/GRPO recompute path (RLinf get_log_prob_value): xs are frozen (the
    latents actually taken), means[k] = mean(x_k, v_theta_new) is differentiable in theta.
    Using the SAME stds as rollout guarantees the importance ratio == 1 before any update.
    """
    per_terms = []
    for k in range(len(means)):
        per_terms.append(_gaussian_logprob(xs[k + 1], means[k], stds[k]))
    stacked = torch.stack(per_terms, dim=1)  # [B,N,L,A]
    if executed_mask is not None:
        m = executed_mask.to(stacked.dtype)
        return (stacked * m).sum(dim=(1, 2, 3))
    return stacked.sum(dim=(1, 2, 3))
