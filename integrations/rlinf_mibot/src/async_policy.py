"""Small, dependency-free policy-version contract for asynchronous actors."""
from __future__ import annotations

from typing import Any


def validate_policy_identity(policy_version: Any, policy_hash: Any) -> tuple[int, str]:
    if isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0:
        raise ValueError(f"policy_version must be a non-negative integer: {policy_version!r}")
    if not isinstance(policy_hash, str) or not policy_hash.strip():
        raise ValueError("policy_hash must be a non-empty string")
    return policy_version, policy_hash.strip()


def validate_policy_lag(*, rollout_version: int, learner_version: int,
                        max_policy_lag: int = 1) -> int:
    if max_policy_lag < 0:
        raise ValueError("max_policy_lag must be non-negative")
    lag = int(learner_version) - int(rollout_version)
    if lag < 0:
        raise ValueError(
            f"rollout policy v{rollout_version} is newer than learner v{learner_version}")
    if lag > max_policy_lag:
        raise ValueError(
            f"stale rollout policy v{rollout_version}: learner=v{learner_version}, "
            f"lag={lag}, max_policy_lag={max_policy_lag}")
    return lag


def validate_trajectory_policy(trajectories: dict, *, policy_version: int,
                               policy_hash: str) -> None:
    """Ensure an episode never crosses an adapter reload boundary."""
    validate_policy_identity(policy_version, policy_hash)
    for trajectory_id, chunks in trajectories.items():
        if not isinstance(trajectory_id, str) or not isinstance(chunks, list) or not chunks:
            raise ValueError("rollout-store trajectory IDs/chunks have invalid structure")
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise ValueError("rollout-store chunks must be dictionaries")
            chunk_version = chunk.get("policy_version", policy_version)
            chunk_hash = chunk.get("policy_hash", policy_hash)
            if chunk_version != policy_version or chunk_hash != policy_hash:
                raise ValueError(
                    f"trajectory {trajectory_id!r} mixes policy snapshots: "
                    f"expected v{policy_version}/{policy_hash}, "
                    f"got v{chunk_version}/{chunk_hash}")
