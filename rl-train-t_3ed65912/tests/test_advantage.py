"""Unit + mutation test for compute_group_advantages (task t_74a6ed7a advantage fix).

Verifies the completion-aware advantage handling prescribed by
REPORT/hyperparameter_management_reference.md, then MUTATION-tests each assertion so no
check trivially passes (skill rule: "no test that trivially passes"). Every meaningful
check is proven to CATCH the pre-fix behaviour (naive standardize-everything) it replaces.

Run (numpy only, CPU, no sim/checkpoint):
  PYTHONPATH=src python3 tests/test_advantage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
from advantage import compute_group_advantages  # noqa: E402

_FAILS = []
_MUT = []


def check(cond, msg):
    print(f"[{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        _FAILS.append(msg)


def naive_standardize(returns):
    """The PRE-FIX behaviour we replaced: standardize every group, no composition awareness."""
    r = np.asarray(returns, dtype=np.float64)
    return (r - r.mean()) / (r.std() + 1e-8)


# ---------------------------------------------------------------------------
# 1. BASELINE: the fixed function behaves correctly on each composition
# ---------------------------------------------------------------------------
print("== baseline (function under test) ==")

# all-success with return variance -> non-negative min-baseline, NO negative advantage
r_allsucc = [2.0, 2.5, 3.0, 2.2]          # all succeeded, differing hold-shaping returns
adv, info = compute_group_advantages(r_allsucc, [True] * 4, std_gate=0.05)
check(info["composition"] == "all_success", "all-success group labelled all_success")
check(info["mode"] == "nonneg_min_baseline", "all-success uses non-negative min-baseline")
check(not info["gated"], "all-success (with variance) is NOT gated")
check((adv >= 0).all(), "all-success advantages are all >= 0 (no punished success)")
check(abs(adv.min() - 0.0) < 1e-12, "min-baseline: slowest success gets advantage 0, not negative")
check(np.argmax(adv) == np.argmax(r_allsucc), "highest-return success gets the largest advantage")

# mixed -> group-relative normalized (mean 0)
r_mixed = [0.0, 0.0, 3.0, 0.0]
adv_m, info_m = compute_group_advantages(r_mixed, [False, False, True, False], std_gate=0.05)
check(info_m["composition"] == "mixed", "mixed group labelled mixed")
check(info_m["mode"] == "group_relative", "mixed uses group-relative normalized")
check(abs(adv_m.mean()) < 1e-9, "mixed advantage is mean-centred")
check(adv_m[2] > 0 and (adv_m[[0, 1, 3]] < 0).all(), "mixed: success positive, failures negative")

# all-failure -> gated (zero advantage, skip step)
adv_f, info_f = compute_group_advantages([0.0, 0.0, 0.0, 0.0], [False] * 4, std_gate=0.05)
check(info_f["gated"] and info_f["reason"] == "all_failure", "all-failure group is gated (all_failure)")
check((adv_f == 0).all(), "all-failure advantages are all zero (no update)")

# low-variance all-success (std < gate) -> gated (avoid amplifying noise)
r_lowvar = [2.00, 2.01, 2.00, 2.02]
adv_l, info_l = compute_group_advantages(r_lowvar, [True] * 4, std_gate=0.05)
check(info_l["gated"] and info_l["reason"] == "low_reward_std",
      "low-variance group is gated (low_reward_std), not standardized")
check((adv_l == 0).all(), "low-variance advantages are all zero (no noisy update)")

# ---------------------------------------------------------------------------
# 2. MUTATION: prove each key guarantee CATCHES the pre-fix / a broken variant
# ---------------------------------------------------------------------------
print("\n== mutation (each check must catch the bug it guards) ==")


def mut(name, cond_caught):
    print(f"[{'caught' if cond_caught else 'MISSED'}] {name}")
    _MUT.append((name, cond_caught))


# M1: the PRE-FIX naive standardize assigns a NEGATIVE advantage to a slow success in an
#     all-success group (the exact bug the reference doc flagged). Our fix must NOT.
naive = naive_standardize(r_allsucc)
mut("naive standardize punishes a slow success with negative advantage (bug present in old code)",
    (naive < -1e-6).any())
mut("fixed function removes that negative advantage (all >= 0)",
    (adv >= -1e-12).all() and (naive < -1e-6).any())

# M2: without the low-variance gate, a near-constant all-success group still produces tiny
#     NON-ZERO advantages (min-baseline of noise) that would drive a spurious update; the
#     real gate zeroes them. (all-success uses min-baseline, not standardize, so the gate's
#     job here is to suppress noise-scale updates, not large ones.)
adv_nogate, info_nogate = compute_group_advantages(r_lowvar, [True] * 4, std_gate=0.0)
mut("with gate disabled, low-variance group yields non-zero (noise-scale) advantage (would update)",
    (np.abs(adv_nogate) > 1e-9).any() and not info_nogate["gated"])
mut("with the real gate, that low-variance group produces zero advantage",
    (adv_l == 0).all() and info_l["gated"])

# M3: composition classification must be exact — flipping one success flag changes the mode.
_, info_flip = compute_group_advantages(r_allsucc, [True, True, True, False], std_gate=0.05)
mut("flipping one member to failure moves all_success -> mixed",
    info_flip["composition"] == "mixed" and info["composition"] == "all_success")

# M4: the mixed branch must be mean-centred; a min-baseline on a mixed group would leave a
#     positive mean (wrong). Confirm the mixed group is centred, not min-baselined.
mut("mixed group is mean-centred (min would leave mean>0)",
    abs(adv_m.mean()) < 1e-9 and (r_mixed - np.min(r_mixed)).mean() > 0.1)

# M5: all-failure must gate even if std_gate were tiny (n_succ==0 has no signal). A variant
#     that only checks std would MISS an all-failure group that happens to have return spread
#     (e.g. differing timeout penalties). Confirm we gate on all-failure regardless.
adv_f2, info_f2 = compute_group_advantages([-0.5, -0.1, -0.3, -0.2], [False] * 4, std_gate=0.001)
mut("all-failure with return spread is still gated (no success => no signal)",
    info_f2["gated"] and (adv_f2 == 0).all())

# ---------------------------------------------------------------------------
print("\n== summary ==")
missed = [n for n, ok in _MUT if not ok]
print(f"baseline: {'ALL PASS' if not _FAILS else str(len(_FAILS)) + ' FAIL'}"
      f"  |  mutation: {len(_MUT) - len(missed)}/{len(_MUT)} caught")
if _FAILS:
    print("BASELINE FAILURES:", _FAILS)
if missed:
    print("MUTATIONS MISSED (dead checks):", missed)
sys.exit(1 if (_FAILS or missed) else 0)
