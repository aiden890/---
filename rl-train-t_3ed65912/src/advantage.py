"""Completion-aware group advantage for GRPO (task t_3ed65912).

Single source of truth for the advantage-handling fix prescribed by
REPORT/hyperparameter_management_reference.md ("Advantage handling before a larger
sweep"). Isolated here (numpy-only, no sim deps) so it is unit- and mutation-testable on
CPU without loading the checkpoint or the simulator. grpo_train_loop imports this.
"""
from __future__ import annotations

import numpy as np


def compute_group_advantages(returns, successes, std_gate=0.05):
    """Composition-aware group advantage.

    The prior code standardized EVERY group with (r-mean)/std. For an all-success group
    that assigns NEGATIVE advantage to the slower successes purely because the group mean
    is higher -- it PENALIZES valid successes, and when the std is tiny it amplifies noise.
    The reference doc prescribes:

      - mixed success/failure -> standard group-relative normalized advantage.
      - all-success           -> Z-1-style completion-aware NON-NEGATIVE min-baseline:
                                 adv = r - min(r) (>= 0). Every success is reinforced;
                                 higher-return (faster/cleaner) successes get MORE; none
                                 are punished for being slower than the group mean.
      - all-failure / effectively-constant reward (std < std_gate) -> GATE: zero advantage,
                                 caller skips the optimizer step (no informative ordering).

    Args:
      returns:   1-D array-like of per-member GRPO returns.
      successes: 1-D bool array-like, per-member trained-skill success.
      std_gate:  minimum within-group reward std to allow an update.

    Returns:
      (adv, info) where adv is a float64 numpy array aligned to `returns` and info carries
      {composition, mode, gated, reason, reward_std, n_success} for run logging.
    """
    r = np.asarray(returns, dtype=np.float64)
    succ = np.asarray(successes, dtype=bool)
    n = len(r)
    std = float(r.std()) if n else 0.0
    n_succ = int(succ.sum())

    if n_succ == 0:
        composition = "all_failure"
    elif n_succ == n:
        composition = "all_success"
    else:
        composition = "mixed"

    # All-failure gate: no success in the group => no positive outcome to reinforce, so skip
    # regardless of any return spread from shaping/timeout penalties (the task rule is
    # "all-failure/무변동 -> skip"; shaping-only ordering is not an approved primary signal).
    if n_succ == 0:
        return np.zeros(n, dtype=np.float64), {
            "composition": composition, "mode": "gated", "gated": True,
            "reason": "all_failure", "reward_std": std, "n_success": 0}

    # Reward-std gate: an effectively-constant-reward group carries no ordering signal, so
    # standardizing it only amplifies numeric noise. Skip it (all-zero advantage).
    if std < std_gate:
        return np.zeros(n, dtype=np.float64), {
            "composition": composition, "mode": "gated", "gated": True,
            "reason": "low_reward_std", "reward_std": std, "n_success": n_succ}

    if composition == "all_success":
        # completion-aware non-negative min-baseline (no negative advantage on a valid success)
        adv = r - r.min()
        mode = "nonneg_min_baseline"
    else:
        # mixed group: standard group-relative normalized advantage
        adv = (r - r.mean()) / (std + 1e-8)
        mode = "group_relative"
    return adv, {"composition": composition, "mode": mode, "gated": False,
                 "reason": None, "reward_std": std, "n_success": n_succ}
