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
from lora import PerSkillLoRALinear, SKILLS, lora_parameters  # noqa: E402

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


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"\n{len(_FAILS)} FAILED" if _FAILS else f"\nAll LoRA tests passed.")
    if _FAILS:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
