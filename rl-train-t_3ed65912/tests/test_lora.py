"""Unit tests for per-skill LoRA (src/lora.py). CPU-only, no model/sim.

Verifies the architectural guarantees the pilot relies on:
  * zero-init B => wrapped Linear == base Linear for EVERY skill at start (pretrained
    policy is the exact eta=0 starting point).
  * only the ACTIVE skill's adapter changes the output (strict isolation).
  * a gradient on skill A's output touches ONLY skill A's params, never B/C's.
  * param bookkeeping: trainable count matches wrappers x skills x 2 x (rank shapes).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from lora import (  # noqa: E402
    PerSkillLoRALinear, SKILLS, inject_per_skill_lora, load_lora_state_dict,
    lora_parameters, lora_state_dict,
)

_FAILS = []


def check(c, m):
    print(("[ok] " if c else "[FAIL] ") + m)
    if not c:
        _FAILS.append(m)


def test_zero_init_is_identity():
    base = nn.Linear(16, 24)
    w = PerSkillLoRALinear(base, SKILLS, rank=8, alpha=32)
    x = torch.randn(2, 5, 16)
    ref = base(x)
    for s in list(SKILLS) + [None]:
        w.set_active_skill(s)
        out = w(x)
        check(torch.allclose(out, ref, atol=1e-6), f"zero-init B: active={s} equals base Linear")


def test_only_active_skill_changes_output():
    torch.manual_seed(0)
    base = nn.Linear(16, 16)
    w = PerSkillLoRALinear(base, SKILLS, rank=4, alpha=16)
    # perturb GRASP's B so its adapter is non-trivial
    with torch.no_grad():
        w.lora_B["grasp"].add_(torch.randn_like(w.lora_B["grasp"]))
    x = torch.randn(1, 3, 16)
    w.set_active_skill("grasp"); g = w(x)
    w.set_active_skill("place"); p = w(x)
    check(not torch.allclose(g, base(x)), "perturbed grasp adapter changes output when active")
    check(torch.allclose(p, base(x), atol=1e-6), "place adapter (zero) leaves output == base (isolation)")


def test_gradient_isolation():
    torch.manual_seed(1)
    base = nn.Linear(12, 12)
    w = PerSkillLoRALinear(base, SKILLS, rank=4, alpha=16)
    x = torch.randn(2, 4, 12)
    w.set_active_skill("grasp")
    w(x).sum().backward()
    # at zero-init B, dL/dA == 0 (B=0 kills A's grad); the meaningful signal is on B.
    grasp_has = w.lora_B["grasp"].grad is not None and w.lora_B["grasp"].grad.abs().sum() > 0
    others_none = all(w.lora_A[s].grad is None and w.lora_B[s].grad is None
                      for s in SKILLS if s != "grasp")
    check(bool(grasp_has), "active skill (grasp) receives gradient (on B at zero-init)")
    check(bool(others_none), "inactive skills (move_holding, place) receive NO gradient (isolation)")
    # base stays frozen
    check(base.weight.grad is None, "frozen base Linear gets no gradient")


def test_param_bookkeeping():
    base = nn.Linear(32, 32)
    w = PerSkillLoRALinear(base, SKILLS, rank=8, alpha=32)
    params = lora_parameters([w])
    # 3 skills x (A[8,32] + B[32,8]) = 3 x 2 tensors
    check(len(params) == len(SKILLS) * 2, "lora_parameters returns A+B per skill")
    only_grasp = lora_parameters([w], skill="grasp")
    check(len(only_grasp) == 2, "per-skill parameter selection returns just that skill's A+B")
    n = sum(p.numel() for p in params)
    check(n == len(SKILLS) * (8 * 32 + 32 * 8), "trainable param count matches rank shapes")


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv_proj = nn.Linear(8, 24)
        self.attn.o_proj = nn.Linear(8, 8)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(8, 16)
        self.mlp.up_proj = nn.Linear(8, 16)
        self.mlp.down_proj = nn.Linear(16, 8)


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = nn.Module()
        self.dit.layers = nn.ModuleList([_Block(), _Block()])
        self.action_projector = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8))
        self.action_output_layer = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
        self.state_projector = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8))
        self.t_embedder = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8))
        self.t_projector = nn.Sequential(nn.Linear(8, 12))
        self.vlm = nn.Sequential(nn.Linear(8, 8))


def test_all_linear_targets_cover_action_expert_and_exclude_vlm():
    model = _Expert()
    wrappers = inject_per_skill_lora(
        model, skills=("grasp",), rank=2, alpha=4, targets=("all_linear",))
    names = [w.target_name for w in wrappers]
    assert len(names) == 19  # 2 blocks * 5 + 2+2+2+2+1 projectors
    assert "dit.layers.0.attn.qkv_proj" in names
    assert "dit.layers.1.attn.o_proj" in names
    assert "dit.layers.0.mlp.gate_proj" in names
    assert "dit.layers.0.mlp.up_proj" in names
    assert "dit.layers.0.mlp.down_proj" in names
    assert any(n.startswith("action_projector.") for n in names)
    assert any(n.startswith("action_output_layer.") for n in names)
    assert any(n.startswith("state_projector.") for n in names)
    assert any(n.startswith("t_embedder.") for n in names)
    assert any(n.startswith("t_projector.") for n in names)
    assert all(not n.startswith("vlm.") for n in names)
    assert isinstance(model.vlm[0], nn.Linear)


def test_named_target_list_matches_wrapped_modules():
    model = _Expert()
    wrappers = inject_per_skill_lora(
        model, skills=("grasp",), rank=2, alpha=4,
        targets=("qkv_proj", "o_proj", "gate_proj"))
    assert [w.target_name for w in wrappers] == [
        "dit.layers.0.attn.qkv_proj", "dit.layers.0.attn.o_proj",
        "dit.layers.0.mlp.gate_proj", "dit.layers.1.attn.qkv_proj",
        "dit.layers.1.attn.o_proj", "dit.layers.1.mlp.gate_proj",
    ]


def test_named_checkpoint_roundtrip_rejects_layout_mismatch():
    source = _Expert()
    source_wrappers = inject_per_skill_lora(
        source, skills=("grasp",), rank=2, alpha=4, targets=("all_linear",))
    with torch.no_grad():
        source_wrappers[3].lora_B["grasp"].fill_(0.25)
    state = lora_state_dict(source_wrappers)
    assert any(key.startswith("dit.layers.0.mlp.up_proj.") for key in state)

    target = _Expert()
    target_wrappers = inject_per_skill_lora(
        target, skills=("grasp",), rank=2, alpha=4, targets=("all_linear",))
    load_lora_state_dict(target_wrappers, state)
    assert torch.equal(
        target_wrappers[3].lora_B["grasp"], source_wrappers[3].lora_B["grasp"])

    wrong_layout = _Expert()
    wrong_wrappers = inject_per_skill_lora(
        wrong_layout, skills=("grasp",), rank=2, alpha=4, targets=("qkv_proj",))
    try:
        load_lora_state_dict(wrong_wrappers, state)
    except RuntimeError:
        pass
    else:
        raise AssertionError("checkpoint layout mismatch must fail")

    wrong_shape = dict(state)
    key = next(iter(wrong_shape))
    wrong_shape[key] = wrong_shape[key][:-1]
    try:
        load_lora_state_dict(target_wrappers, wrong_shape)
    except RuntimeError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("checkpoint tensor shape mismatch must fail")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"\n{len(_FAILS)} FAILED" if _FAILS else f"\nAll LoRA tests passed.")
    if _FAILS:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
