"""Adapter: run the flow-SDE sampler on the real Xiaomi MiBoT action expert.

The checkpoint's `MiBoTForActionGeneration.forward` (modeling_mibot.py) does two things:

  1. CONDITIONING PREP (expensive, run once): a VLM forward pass over the video/text,
     then builds `position_embeds`, `attn_mask`, and `state_embed`, and closes over a
     `dit_forward_fn(noisy_action, t)` -> velocity.
  2. INTEGRATION (cheap, N steps): the deterministic Euler loop
        x <- x + dit_forward_fn(x, t) * dt.

For RL we keep (1) byte-identical to the checkpoint and replace only (2) with the
flow-SDE integrator from flow_sde.py, which records per-transition log-probs.

`build_velocity_field(model, **model_kwargs)` reproduces the prep from the checkpoint's
forward and returns `(velocity_field, shape)`. It is the ONE place that mirrors upstream
internals; it is small, documented, and asserted against the model's own deterministic
output by scripts/sde_probe.py (eta=0 must equal model.forward().actions bit-for-bit).
If upstream changes its forward, that probe fails loudly rather than drifting silently.

We do NOT edit or copy the checkpoint modeling file (it is read-only and hash-pinned);
we reach into the loaded module's public attributes (`dit_forward`, `state_projector`,
`rotary_emb`, `t_embedder`, etc.), exactly the objects the checkpoint's own forward uses.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from flow_sde import FlowSDEResult, flow_sde_sample, make_executed_mask  # noqa: E402


def build_velocity_field(model: Any, state: torch.Tensor, action_mask: torch.Tensor, **kwargs):
    """Recreate the checkpoint forward's conditioning prep and return a velocity field.

    Mirrors MiBoTForActionGeneration.forward up to (but not including) the Euler loop.
    Returns (velocity_field, shape, device, dtype) where velocity_field(x, t) == the
    checkpoint's own `dit_forward_fn(x, t)`.
    """
    with torch.no_grad():
        vlm_outputs = model.vlm(**kwargs, use_cache=True)

    action_bs, action_length, _ = action_mask.shape
    _, state_length, _ = state.shape
    dit_query_length = action_length + state_length + 1

    position_ids = (
        torch.arange(0, dit_query_length, device=action_mask.device).view(1, 1, -1).repeat(3, action_bs, 1)
        + vlm_outputs.position_ids.max(dim=-1)[0][..., None]
        + 1
    )
    position_embeds = model.rotary_emb(action_mask, position_ids)

    dit_mask = torch.tril(
        torch.ones((action_bs, dit_query_length, dit_query_length), device=action_mask.device), diagonal=0
    )
    cache_mask = vlm_outputs.attention_mask[:, None, :].expand(-1, dit_query_length, -1)
    attn_mask = torch.cat([cache_mask, dit_mask], dim=-1)[:, None].bool()

    state_embed = model.state_projector(state)
    past_key_values = vlm_outputs.past_key_values

    def velocity_field(noisy_action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return model.dit_forward(
            noisy_action=noisy_action,
            t=t,
            action_mask=action_mask,
            state_embed=state_embed,
            position_embeds=position_embeds,
            past_key_values=past_key_values,
            attn_mask=attn_mask,
        )

    shape = (action_bs, action_length, action_mask.shape[-1])
    return velocity_field, shape, action_mask.device, action_mask.dtype


def sample_action_chunk_sde(
    model: Any,
    state: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    num_steps: int = 5,
    eta: float = 0.0,
    replan_steps: int | None = None,
    real_action_dim: int | None = None,
    generator: torch.Generator | None = None,
    x0: torch.Tensor | None = None,
    grad: bool = False,
    **kwargs,
) -> FlowSDEResult:
    """Draw one action chunk (a group if action_mask batch>1) via the flow-SDE.

    eta=0 -> deterministic, equals the checkpoint's Euler sampler.
    eta>0 -> stochastic policy with per-transition log-probs for GRPO.

    `grad=True` keeps autograd through the velocity field (used only when recomputing
    on the training GPU; rollout uses grad=False for speed).
    """
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        velocity_field, shape, device, dtype = build_velocity_field(model, state, action_mask, **kwargs)
        B, L, A = shape
        executed_mask = None
        if replan_steps is not None:
            executed_mask = make_executed_mask(L, A, replan_steps, real_action_dim, device=device)
        result = flow_sde_sample(
            velocity_field,
            shape,
            num_steps=num_steps,
            eta=eta,
            device=device,
            dtype=dtype,
            generator=generator,
            executed_mask=executed_mask,
            x0=x0,
        )
    return result
