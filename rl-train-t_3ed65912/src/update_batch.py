"""Pure helpers for multi-group GRPO update batching."""
from __future__ import annotations

import copy
import struct
from dataclasses import dataclass, field


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


class IdempotentUpdateCache:
    """Replay a completed logical update result without executing its operation twice."""

    def __init__(self):
        self.completed = {}
        self.failed = {}

    def run(self, update_id, operation):
        if not update_id:
            raise ValueError("update_id is required")
        if update_id in self.failed:
            raise RuntimeError(
                f"update_id {update_id!r} is poisoned after uncertain failure: "
                f"{self.failed[update_id]}")
        if update_id not in self.completed:
            try:
                self.completed[update_id] = copy.deepcopy(operation())
            except Exception as error:
                self.failed[update_id] = repr(error)
                raise
        return copy.deepcopy(self.completed[update_id])


def retry_idempotent_rpc(send, reconnect, request):
    """Retry one update RPC only for transport loss, preserving its logical update ID."""
    try:
        return send(request)
    except (OSError, EOFError, struct.error):
        reconnect()
        return send(request)


def recv_exact(sock, size):
    """Receive exactly ``size`` bytes or fail instead of spinning on EOF."""
    data = bytearray()
    while len(data) < int(size):
        packet = sock.recv(int(size) - len(data))
        if not packet:
            raise EOFError(
                f"connection closed after {len(data)} of {int(size)} response bytes")
        data.extend(packet)
    return bytes(data)


@dataclass
class BatchAccumulator:
    """Validate and aggregate complete groups without renormalizing advantages."""

    advantages: dict[str, float] = field(default_factory=dict)
    groups: list[dict] = field(default_factory=list)
    group_env_seeds: list[int] = field(default_factory=list)
    stored_chunks: int = 0
    trainable_chunks: int = 0

    def add_group(self, env_seed, group_index, advantages, chunks_by_traj, summary):
        env_seed = int(env_seed)
        if env_seed in self.group_env_seeds:
            raise RuntimeError(f"duplicate env seed in update batch: {env_seed}")
        group_advantages = {str(key): float(value) for key, value in advantages.items()}
        duplicate = set(group_advantages) & set(self.advantages)
        if duplicate:
            raise RuntimeError(f"duplicate trajectory ids in update batch: {sorted(duplicate)}")

        expected = set(self.advantages) | set(group_advantages)
        chunks = {str(key): int(value) for key, value in chunks_by_traj.items()}
        actual = set(chunks)
        if actual != expected:
            raise RuntimeError(
                "store trajectory mismatch after group collection: "
                f"missing={sorted(expected - actual)} stale={sorted(actual - expected)}")
        if any(value < 1 for value in chunks.values()):
            raise RuntimeError("every collected trajectory must contain at least one stored chunk")

        group_stored = sum(chunks[trajectory_id] for trajectory_id in group_advantages)
        group_trainable = sum(
            chunks[trajectory_id]
            for trajectory_id, advantage in group_advantages.items()
            if advantage != 0.0)
        item = dict(summary)
        item.update({
            "group_index": int(group_index),
            "env_seed": env_seed,
            "trajectory_ids": list(group_advantages),
            "advantages": group_advantages,
            "stored_chunks": group_stored,
            "trainable_chunks": group_trainable,
        })
        self.advantages.update(group_advantages)
        self.groups.append(item)
        self.group_env_seeds.append(env_seed)
        self.stored_chunks = sum(chunks.values())
        self.trainable_chunks += group_trainable
        return item


def execute_batched_update(*, update_index, min_groups, min_trainable_chunks, max_groups,
                           select_seed, collect_group, get_store, update, reset_store,
                           on_group=None):
    """Collect validated whole groups and invoke exactly one logical update RPC."""
    validate_batch_config(min_groups, min_trainable_chunks, max_groups)
    accumulator = BatchAccumulator()
    update_calls = 0
    try:
        initial = get_store()
        if int(initial.get("n_chunks", 0)) != 0:
            raise RuntimeError(f"trainer store not empty at update start: {initial}")
        while needs_more_groups(len(accumulator.groups), accumulator.trainable_chunks,
                                min_groups, min_trainable_chunks):
            group_index = len(accumulator.groups)
            if group_index >= int(max_groups):
                raise RuntimeError(
                    f"chunk floor not reached after {max_groups} groups: "
                    f"{accumulator.trainable_chunks} < {min_trainable_chunks}")
            selected = select_seed(update_index, group_index, set(accumulator.group_env_seeds))
            env_seed, source = selected if isinstance(selected, tuple) else (selected, "raw")
            iteration_token = batch_iteration_token(update_index, group_index)
            summary = collect_group(group_index, int(env_seed), iteration_token)
            if "_advantages" not in summary:
                raise RuntimeError(
                    f"batched group did not produce advantages: {summary.get('skipped', summary)}")
            group_advantages = summary.pop("_advantages")
            group = accumulator.add_group(
                env_seed, group_index, group_advantages,
                get_store().get("chunks_by_traj", {}), summary)
            group["seed_source"] = source
            if on_group is not None:
                on_group(group, accumulator)

        update_calls += 1
        result = update(dict(accumulator.advantages))
        result.update({
            "groups_collected": len(accumulator.groups),
            "stored_chunks_collected": accumulator.stored_chunks,
            "trainable_chunks_collected": accumulator.trainable_chunks,
            "group_env_seeds": list(accumulator.group_env_seeds),
            "group_summaries": accumulator.groups,
            "advantages": dict(accumulator.advantages),
            "stored_trajectory_ids": list(accumulator.advantages),
            "update_rpc_calls": update_calls,
        })
        return result
    except Exception:
        reset_store()
        raise


def audit_batch_trace(trace):
    """Fail-closed static/mutation audit of a serialized multi-group update trace."""
    groups = list(trace.get("group_summaries", []))
    seeds = list(trace.get("group_env_seeds", []))
    group_ids = [str(item) for group in groups for item in group.get("trajectory_ids", [])]
    aggregate = {str(key): float(value) for key, value in trace.get("advantages", {}).items()}
    group_advantages = {
        str(key): float(value)
        for group in groups
        for key, value in group.get("advantages", {}).items()
    }
    checks = {
        "all_groups_present": int(trace.get("groups_collected", -1)) == len(groups),
        "distinct_group_seeds": bool(seeds) and len(seeds) == len(groups) == len(set(seeds)),
        "group_local_advantages_preserved": group_advantages == aggregate,
        "exactly_one_update_rpc": int(trace.get("update_rpc_calls", 0)) == 1,
        "distinct_trajectory_ids": len(group_ids) == len(set(group_ids)),
        "store_complete": set(map(str, trace.get("stored_trajectory_ids", []))) == set(group_ids),
        "elementwise_nonjoint_ratio": trace.get("joint_logprob") is False,
        "ratio_mask_shape_match": trace.get("ratio_shape") == trace.get("mask_shape"),
    }
    return {"pass": all(checks.values()), "checks": checks}
