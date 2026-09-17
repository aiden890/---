"""Per-skill LoRA adapters + termination head for the MiBoT DiT action expert.

Architecture (task t_3ed65912, per operator direction + research brief rl_finetune_brief.md):
  * VLM backbone: FROZEN.
  * DiT action-expert body + projectors: FROZEN.
  * TRAINABLE = one LoRA adapter set PER SKILL (GRASP / MOVE_HOLDING / PLACE) injected
    into configured action-expert Linear layers (qkv-only legacy or all-linear), plus an optional
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
        self.target_name = ""
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
    collect the LoRA parameters. ``targets=("all_linear",)`` covers every Linear in the
    action expert: DiT attention/MLP plus state/action/time input and action output
    projectors. The VLM is excluded by an explicit prefix allowlist.
    """
    targets = tuple(targets)
    all_linear = targets == ("all_linear",)
    target_leaves = set(targets)
    projector_prefixes = (
        "action_projector.", "action_output_layer.", "state_projector.",
        "t_embedder.", "t_projector.", "sink.",
    )

    selected = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        in_dit = name.startswith("dit.layers.")
        if all_linear:
            if in_dit or any(name.startswith(prefix) for prefix in projector_prefixes):
                selected.append((name, module))
        elif in_dit and name.rsplit(".", 1)[-1] in target_leaves:
            selected.append((name, module))

    wrappers: list[PerSkillLoRALinear] = []
    for name, base in selected:
        parent_name, leaf = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        wrapped = PerSkillLoRALinear(base, skills, rank=rank, alpha=alpha)
        wrapped.target_name = name
        wrapped.to(device=base.weight.device, dtype=base.weight.dtype)
        setattr(parent, leaf, wrapped)
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


def lora_state_dict(wrappers: Iterable[PerSkillLoRALinear]):
    """Serialize adapters by stable target-module name, never wrapper-list position."""
    state = {}
    for w in wrappers:
        if not w.target_name:
            raise RuntimeError("LoRA wrapper is missing target_name")
        for skill in w.skills:
            state[f"{w.target_name}.lora_A.{skill}"] = w.lora_A[skill].detach().cpu()
            state[f"{w.target_name}.lora_B.{skill}"] = w.lora_B[skill].detach().cpu()
    return state


def load_lora_state_dict(wrappers: Iterable[PerSkillLoRALinear], state):
    """Strictly restore a named adapter layout; fail on missing or unexpected tensors."""
    wrappers = list(wrappers)
    expected = set(lora_state_dict(wrappers))
    actual = set(state)
    if expected != actual:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(
            f"LoRA checkpoint layout mismatch: missing={missing[:3]} "
            f"unexpected={unexpected[:3]}")
    for w in wrappers:
        for skill in w.skills:
            for kind, params in (("lora_A", w.lora_A), ("lora_B", w.lora_B)):
                key = f"{w.target_name}.{kind}.{skill}"
                if tuple(state[key].shape) != tuple(params[skill].shape):
                    raise RuntimeError(
                        f"LoRA checkpoint shape mismatch for {key}: "
                        f"expected={tuple(params[skill].shape)} actual={tuple(state[key].shape)}")
    with torch.no_grad():
        for w in wrappers:
            for skill in w.skills:
                for kind, params in (("lora_A", w.lora_A), ("lora_B", w.lora_B)):
                    key = f"{w.target_name}.{kind}.{skill}"
                    params[skill].copy_(state[key].to(
                        device=params[skill].device, dtype=params[skill].dtype))


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
