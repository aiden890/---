"""Crash-safe state and validation for double-buffered actor/learner training.

This module deliberately contains no SSH or GPU code.  The production launcher uses
these helpers, while unit tests exercise every state transition without requiring a
RoboCasa installation.
"""
from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


STATE_SCHEMA = "mibot-async-training-v1"
BATCH_SCHEMA = "mibot-rollout-batch-v1"


class AmbiguousUpdateError(RuntimeError):
    """Raised when retrying could apply an optimizer update twice."""


def snapshot(version: int, policy_hash: str) -> dict[str, Any]:
    version = int(version)
    policy_hash = str(policy_hash)
    if version < 0 or not policy_hash:
        raise ValueError(f"invalid policy snapshot: v{version} {policy_hash!r}")
    return {"version": version, "hash": policy_hash}


def snapshot_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return snapshot(metrics["policy_version"], metrics["policy_hash"])


def same_snapshot(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (int(left["version"]), str(left["hash"])) == (
        int(right["version"]), str(right["hash"]))


def make_batch(batch_id: str, expected_snapshot: dict[str, Any], attempt: int) -> dict[str, Any]:
    """Create an adaptive collection contract for one logical update."""
    batch_id = str(batch_id)
    if not batch_id or "/" in batch_id or batch_id in {".", ".."}:
        raise ValueError(f"unsafe batch id: {batch_id!r}")
    expected_snapshot = snapshot(expected_snapshot["version"], expected_snapshot["hash"])
    return {
        "schema": BATCH_SCHEMA,
        "batch_id": batch_id,
        "attempt": int(attempt),
        "policy_snapshot": expected_snapshot,
        "target_trainable_action_chunks": 1024,
        "trainable_action_chunks_estimate": 0,
        "rollout_epochs": [],
        "status": "planned",
    }


def validate_batch(batch: dict[str, Any], *, require_ready: bool = True) -> None:
    if batch.get("schema") != BATCH_SCHEMA:
        raise ValueError(f"unsupported rollout batch schema: {batch.get('schema')!r}")
    snapshot(batch["policy_snapshot"]["version"], batch["policy_snapshot"]["hash"])
    epochs = list(batch.get("rollout_epochs", []))
    indices = [int(item["index"]) for item in epochs]
    if not indices or indices != list(range(len(indices))):
        raise ValueError(f"rollout epochs must be non-empty and contiguous from zero, got {indices}")
    run_ids = [str(item["run_id"]) for item in epochs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("each rollout epoch must use a distinct run id")
    target = int(batch.get("target_trainable_action_chunks", 0))
    actual = int(batch.get("trainable_action_chunks_estimate", 0))
    if target <= 0 or actual < target:
        raise ValueError(f"adaptive rollout target not met: {actual} < {target}")
    if require_ready and (batch.get("status") != "ready" or
                          any(item.get("status") != "ready" for item in epochs)):
        raise ValueError("rollout batch is not durably ready")


def validate_rows_snapshot(rows: list[dict[str, Any]], expected: dict[str, Any]) -> None:
    """Reject an epoch containing missing, mixed, or unexpected policy identities."""
    if not rows:
        raise ValueError("rollout epoch contains no episode rows")
    identities = {
        (int(row["trainer_payload"]["policy_version"]),
         str(row["trainer_payload"]["policy_hash"]))
        for row in rows
    }
    wanted = (int(expected["version"]), str(expected["hash"]))
    if identities != {wanted}:
        raise ValueError(f"rollout policy snapshot mismatch: {sorted(identities)} != {wanted}")


def validate_lag(batch: dict[str, Any], learner: dict[str, Any], max_lag: int = 1) -> int:
    validate_batch(batch)
    lag = int(learner["version"]) - int(batch["policy_snapshot"]["version"])
    if lag < 0 or lag > int(max_lag):
        raise ValueError(
            f"rollout policy lag must be in [0, {max_lag}], got {lag}: "
            f"rollout=v{batch['policy_snapshot']['version']} learner=v{learner['version']}")
    return lag


def validate_update_report(report: dict[str, Any], batch: dict[str, Any],
                           learner_before: dict[str, Any]) -> dict[str, Any]:
    """Validate the optimizer commit, checkpoint, and exact source snapshot."""
    validate_batch(batch)
    if report.get("status") != "PASS":
        raise ValueError(f"learner gate did not pass: {report.get('status')!r}")
    if not report.get("checkpoint_reloadable") or not report.get("checkpoint"):
        raise ValueError("accepted update has no reloadable checkpoint")
    update = report.get("update", {})
    rollout_version = int(update["rollout_policy_version"])
    before = int(update["policy_version_before"])
    after = int(update["policy_version_after"])
    expected_rollout = int(batch["policy_snapshot"]["version"])
    expected_before = int(learner_before["version"])
    if rollout_version != expected_rollout:
        raise ValueError(
            f"learner used rollout v{rollout_version}, expected v{expected_rollout}")
    if before != expected_before or after != before + 1:
        raise ValueError(
            f"optimizer version transition must be v{expected_before}->v{expected_before + 1}, "
            f"got v{before}->v{after}")
    expected_lag = expected_before - expected_rollout
    if int(update["policy_lag"]) != expected_lag or expected_lag not in (0, 1):
        raise ValueError("learner reported an invalid policy lag")
    if update.get("replayed_update"):
        # A terminal report is only written for the first successful call.  Treating a
        # replay as newly accepted would advance coordinator accounting twice.
        raise ValueError("replayed optimizer response cannot create a new accepted update")
    result = snapshot(after, update["policy_hash_after"])
    return result


def recovery_action(state: dict[str, Any], learner_now: dict[str, Any],
                    report_exists: bool) -> str:
    """Choose a restart action without ever risking a duplicate optimizer step."""
    inflight = state.get("inflight")
    if not inflight:
        return "idle"
    before = inflight["learner_before"]
    if report_exists:
        return "commit_report"
    if same_snapshot(snapshot_from_metrics(learner_now), before):
        return "retry_same_update_id"
    if int(learner_now["policy_version"]) > int(before["version"]):
        raise AmbiguousUpdateError(
            "learner advanced without a terminal report; refusing a retry that could "
            "duplicate the optimizer update")
    raise RuntimeError("learner snapshot regressed or changed unexpectedly during recovery")


def new_state(actor: dict[str, Any], learner: dict[str, Any], target_updates: int) -> dict[str, Any]:
    actor = snapshot_from_metrics(actor)
    learner = snapshot_from_metrics(learner)
    if not same_snapshot(actor, learner):
        raise ValueError(f"initial actor/learner mismatch: {actor} != {learner}")
    return {
        "schema": STATE_SCHEMA,
        "target_updates": int(target_updates),
        "accepted_updates": 0,
        "next_attempt": 1,
        "actor": actor,
        "learner": learner,
        "ready": None,
        "inflight": None,
        "history": [],
    }


class StateStore:
    """Atomic JSON state file with a single-coordinator process lock."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def load(self) -> dict[str, Any]:
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if value.get("schema") != STATE_SCHEMA:
            raise ValueError(f"unsupported coordinator state: {value.get('schema')!r}")
        return value

    def save(self, value: dict[str, Any]) -> None:
        if value.get("schema") != STATE_SCHEMA:
            raise ValueError("refusing to save malformed coordinator state")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        data = json.dumps(value, indent=2, sort_keys=True) + "\n"
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(self.path))
        directory_fd = os.open(str(self.path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @contextmanager
    def locked(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("another asynchronous coordinator is already running") from exc
            yield
