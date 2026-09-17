from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from update_batch import (  # noqa: E402
    batch_iteration_token, group_env_seed, needs_more_groups, trajectory_action_seed,
    validate_batch_config,
)


def test_collects_minimum_groups_even_when_chunk_target_is_met():
    assert needs_more_groups(groups=7, chunks=2000, min_groups=8, min_chunks=1024)
    assert not needs_more_groups(groups=8, chunks=2000, min_groups=8, min_chunks=1024)


def test_collects_extra_whole_groups_until_chunk_target_is_met():
    assert needs_more_groups(groups=8, chunks=900, min_groups=8, min_chunks=1024)
    assert not needs_more_groups(groups=9, chunks=1050, min_groups=8, min_chunks=1024)


def test_group_seeds_are_distinct_across_and_within_updates():
    seeds = {
        group_env_seed(seed_base=1000, update_index=update, group_index=group)
        for update in range(3)
        for group in range(12)
    }
    assert len(seeds) == 36
    assert group_env_seed(1000, 0, 0) == 1000


def test_seed_namespaces_do_not_collide_past_old_fixed_strides():
    assert group_env_seed(1000, 0, 10_000) != group_env_seed(1000, 1, 0)
    tokens = {batch_iteration_token(u, g) for u in range(4) for g in range(120)}
    assert len(tokens) == 480
    action_seeds = {
        trajectory_action_seed(1000, token, member)
        for token in tokens
        for member in range(3)
    }
    assert len(action_seeds) == 1440


def test_batch_config_rejects_invalid_floors_and_caps():
    validate_batch_config(min_groups=8, min_chunks=1024, max_groups=24)
    for values in [(0, 1024, 24), (8, -1, 24), (8, 1024, 7)]:
        try:
            validate_batch_config(*values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid batch config: {values}")
