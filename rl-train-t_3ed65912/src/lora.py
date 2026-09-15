"""Per-skill LoRA adapters + termination head for the MiBoT DiT action expert.

Architecture (task t_3ed65912, per operator direction + research brief rl_finetune_brief.md):
  * VLM backbone: FROZEN.
  * DiT action-expert body + projectors: FROZEN.
  * TRAINABLE = one LoRA adapter set PER SKILL (GRASP / MOVE_HOLDING / PLACE) injected
    into every DiT layer's attention qkv_proj (and optionally o_proj), plus an optional
    per-skill termination head. Only the ACTIVE skill's adapter contributes to the
    forward pass, so each skill's adapter is trained ONLY on that skill's action
    segments -- strict parameter isolation, zero cross-skill interference (CORAL,
    arXiv:2603.09298). eta=0 + all adapters zero-init => bit-for-bit the pretrained
    policy (B init = 0), so the pretrained behaviour is the exact starting point.

Research-grounded caveats encoded here:
  * LoRA (not full-body) keeps the optimizer state tiny (~few M params/skill) so it fits
    beside the resident inference server -- the full-body probe showed AdamW OOMs.
  * Per-skill LoRA on a DiT action-expert has no *direct* published precedent (CORAL does
    per-task LoRA on the VLM, not the DiT); this is a documented extrapolation. The safer
    fallback (single shared LoRA + skill-id token) is a config switch: use one skill key.
  * PLACE is data-starved (short/rare); the pilot trains GRASP first (curriculum), the
    other adapters are wired but stay at zero-init until they get data.

No checkpoint file is modified: we wrap the loaded nn.Linear modules in place.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn

SKILLS = ("grasp", "move_holding", "place")


class PerSkillLoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with a dict of per-skill low-rank adapters.

    y = W0 x + b0 + (alpha/r) * B_skill(A_skill(x))     [only the active skill]

    B is zero-initialised so the wrapped module == the base Linear at start (for every
    skill), making the pretrained policy the exact eta=0 starting point.
    """

    def __init__(self, base: nn.Linear, skills: Iterable[str], rank: int = 8, alpha: int = 32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.skills = tuple(skills)
        self.active_skill: Optional[str] = None
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for s in self.skills:
            A = nn.Parameter(torch.empty(rank, self.in_features))
            B = nn.Parameter(torch.zeros(self.out_features, rank))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            self.lora_A[s] = A
            self.lora_B[s] = B

    def set_active_skill(self, skill: Optional[str]):
        self.active_skill = skill

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        s = self.active_skill
        if s is None or s not in self.lora_A:
            return out
        A = self.lora_A[s].to(x.dtype)
        B = self.lora_B[s].to(x.dtype)
        delta = torch.nn.functional.linear(torch.nn.functional.linear(x, A), B)
        return out + self.scaling * delta


class SkillTerminationHead(nn.Module):
    """Per-skill 2-class head: from a pooled DiT hidden state -> P(terminate skill).

    Optional; trained from the skill monitor's ground-truth transition events. Kept
    small (hidden->2) and per-skill so it never interferes across skills.
    """

    def __init__(self, hidden_size: int, skills: Iterable[str]):
        super().__init__()
        self.heads = nn.ModuleDict({s: nn.Linear(hidden_size, 2) for s in skills})

    def forward(self, pooled: torch.Tensor, skill: str) -> torch.Tensor:
        return self.heads[skill](pooled)


def inject_per_skill_lora(model, skills=SKILLS, rank: int = 8, alpha: int = 32,
                          targets=("qkv_proj",)) -> list[PerSkillLoRALinear]:
    """Replace targeted DiT-layer Linear submodules with PerSkillLoRALinear wrappers.

    Returns the list of injected wrappers so the caller can set the active skill and
    collect the LoRA parameters. Only touches `model.dit.layers[*].attn.<target>`.
    """
    wrappers: list[PerSkillLoRALinear] = []
    dit = model.dit
    for layer in dit.layers:
        attn = layer.attn
        for tname in targets:
            base = getattr(attn, tname)
            if isinstance(base, PerSkillLoRALinear):
                continue
            wrapped = PerSkillLoRALinear(base, skills, rank=rank, alpha=alpha)
            wrapped.to(device=base.weight.device, dtype=base.weight.dtype)
            setattr(attn, tname, wrapped)
            wrappers.append(wrapped)
    return wrappers


def set_active_skill(wrappers: Iterable[PerSkillLoRALinear], skill: Optional[str]):
    for w in wrappers:
        w.set_active_skill(skill)


def lora_parameters(wrappers: Iterable[PerSkillLoRALinear], skill: Optional[str] = None):
    """Trainable LoRA params, optionally restricted to one skill (for per-skill opt)."""
    params = []
    for w in wrappers:
        keys = [skill] if skill is not None else list(w.skills)
        for s in keys:
            params.append(w.lora_A[s])
            params.append(w.lora_B[s])
    return params


def freeze_all_but_lora(model, wrappers, termination: Optional[SkillTerminationHead] = None):
    """VLM + DiT body + projectors frozen; only LoRA (+ optional termination) trainable."""
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for w in wrappers:
        for p in list(w.lora_A.parameters()) + list(w.lora_B.parameters()):
            p.requires_grad_(True)
            n += p.numel()
    if termination is not None:
        for p in termination.parameters():
            p.requires_grad_(True)
            n += p.numel()
    return n


# The action-expert's I/O projection layers around the frozen 36-layer DiT body.
# "Selective joint training of the action expert" (Z-1, arXiv:2606.31846) opens these
# while keeping the heavy DiT body frozen -- memory-safe (SGD has no extra optimizer
# state) and genuinely trains the expert beyond the attention LoRA.
EXPERT_PREFIXES = ("action_projector.", "action_output_layer.", "state_projector.",
                   "t_embedder.", "t_projector.", "sink.")
# Top-of-stack VLM pieces (smallest slice with grad) for the arm-C selective-joint probe.
VLM_SUFFIXES = ("vlm.model.language_model.norm.",)


def select_trainable(model, wrappers, mode: str, termination: Optional[SkillTerminationHead] = None):
    """Set requires_grad per ablation arm. Returns (n_trainable, groups dict).

    mode:
      adapter_only          -> per-skill LoRA only (arm A; user's 1st direction).
      adapter_plus_expert    -> LoRA + action-expert projection layers (arm B; Z-1 selective joint).
      adapter_plus_expert_vlm-> arm B + a small VLM slice with grad (arm C; best-effort, may OOM).
    """
    n = freeze_all_but_lora(model, wrappers, termination)
    groups = {"lora": n}
    if mode in ("adapter_plus_expert", "adapter_plus_expert_vlm"):
        ne = 0
        for name, p in model.named_parameters():
            if any(name.startswith(pre) for pre in EXPERT_PREFIXES):
                p.requires_grad_(True); ne += p.numel()
        groups["expert_proj"] = ne; n += ne
    if mode == "adapter_plus_expert_vlm":
        nv = 0
        for name, p in model.named_parameters():
            if any(suf in name for suf in VLM_SUFFIXES):
                p.requires_grad_(True); nv += p.numel()
        groups["vlm_slice"] = nv; n += nv
    return n, groups
