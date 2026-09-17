"""Pure helpers for multi-group GRPO update batching."""
from __future__ import annotations


def _pair_nonnegative(a: int, b: int) -> int:
    """Cantor pairing: injective for every pair of non-negative integers."""
    a, b = int(a), int(b)
    if a < 0 or b < 0:
        raise ValueError(f"pair inputs must be non-negative, got {(a, b)}")
    s = a + b
    return s * (s + 1) // 2 + b


def validate_batch_config(min_groups: int, min_chunks: int, max_groups: int) -> None:
    min_groups, min_chunks, max_groups = int(min_groups), int(min_chunks), int(max_groups)
    if min_groups < 1:
        raise ValueError("groups_per_update must be >= 1")
    if min_chunks < 0:
        raise ValueError("min_trainable_chunks must be >= 0")
    if max_groups < min_groups:
        raise ValueError("max_groups_per_update must be >= groups_per_update")


def needs_more_groups(groups: int, chunks: int, min_groups: int, min_chunks: int) -> bool:
    """Collect whole groups until both the group and trainable-chunk floors are met."""
    return int(groups) < int(min_groups) or int(chunks) < int(min_chunks)


def batch_iteration_token(update_index: int, group_index: int) -> int:
    """Unique namespace token for a group across all update/group indices."""
    return _pair_nonnegative(update_index, group_index)


def group_env_seed(seed_base: int, update_index: int, group_index: int) -> int:
    """Deterministic, collision-free env seed for one group in an update batch."""
    return int(seed_base) + batch_iteration_token(update_index, group_index)


def trajectory_action_seed(seed_base: int, iteration_token: int, member_index: int) -> int:
    """Unique action-noise seed for a trajectory member."""
    return int(seed_base) + _pair_nonnegative(iteration_token, member_index)
