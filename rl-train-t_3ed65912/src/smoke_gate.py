"""Artifact-backed gate for the two-update all-linear GRASP smoke."""
from __future__ import annotations

import math
from typing import Any


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_all_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(v) for v in value)
    return True


def evaluate_smoke(rows):
    rows = list(rows)
    seeds = [int(seed) for row in rows for seed in row.get("group_env_seeds", [])]
    epoch0 = [row.get("epoch0_mean_ratio") for row in rows]
    logprob_diffs = [
        float((row.get("epoch_stats") or [{}])[0].get("mean_abs_logratio", float("inf")))
        for row in rows
    ]
    checks = {
        "two_updates": len(rows) == 2,
        "at_least_128_trajectories": sum(int(r.get("trajectories_collected", 0)) for r in rows) >= 128,
        "groups_per_update_at_least_8": bool(rows) and all(
            int(r.get("groups_collected", 0)) >= 8 for r in rows),
        "trainable_chunks_per_update_at_least_1024": bool(rows) and all(
            int(r.get("trainable_chunks_collected", 0)) >= 1024 for r in rows),
        "distinct_group_env_seeds": bool(seeds) and len(seeds) == len(set(seeds)),
        "epoch0_ratio_one": len(epoch0) == len(rows) and all(
            isinstance(v, (int, float)) and abs(float(v) - 1.0) <= 1e-6 for v in epoch0),
        "rollout_recompute_logprob_match": len(logprob_diffs) == len(rows) and all(
            v <= 1e-6 for v in logprob_diffs),
        "no_nonfinite": all(_all_finite(r) for r in rows) and all(
            int(r.get("post_step_n_nonfinite", 0)) == 0 and
            all(int(e.get("n_nonfinite", 0)) == 0 for e in r.get("epoch_stats", []))
            for r in rows),
        "adapter_changed": bool(rows) and all(float(r.get("adapter_delta_l2", 0.0)) > 0 for r in rows),
        "no_oom": len(rows) == 2,
    }
    return {"pass": all(checks.values()), "checks": checks}
