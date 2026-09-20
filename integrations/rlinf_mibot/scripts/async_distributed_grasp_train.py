#!/usr/bin/env python3
"""Resumable double-buffered DAX rollout / RTX 3090 GRPO coordinator."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import distributed_grasp_train as legacy  # noqa: E402
from consume_collector import (  # noqa: E402
    build_smoke_advantages,
    filter_homogeneous_groups,
)
from async_training_coordinator import (  # noqa: E402
    AmbiguousUpdateError,
    StateStore,
    make_batch,
    new_state,
    recovery_action,
    same_snapshot,
    snapshot_from_metrics,
    validate_batch,
    validate_lag,
    validate_rows_snapshot,
    validate_update_report,
)


def _remote_stream(mode: str, batch_id: str, *extra: str) -> dict:
    command = (
        "docker run --rm --network host -e PYTHONPATH=/integration/src "
        f"-v {legacy.LEARNER_ROOT}/integration:/integration:ro "
        f"-v {legacy.LEARNER_ROOT}/results:/results xiaomi-cu121:t_9f03a613 "
        f"python3 /integration/src/streaming_consume.py {mode} /results/{batch_id} "
        "--host 127.0.0.1 --port 10088 "
        "--config /integration/configs/consume_production.yaml " + " ".join(extra)
    )
    result = legacy.ssh(legacy.LEARNER, command, capture=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def estimate_trainable_chunks(rows: list[dict], epoch_index: int) -> dict:
    """Estimate the exact trainer gate from signed collector metadata."""
    items = []
    chunks_by_trajectory = {}
    for row in rows:
        payload = row["trainer_payload"]
        trajectory_ids = list(map(str, payload["trajectory_ids"]))
        chunks = int(payload.get("n_chunks", 0))
        if chunks < 0 or not trajectory_ids:
            raise ValueError("invalid collector chunk metadata")
        if len(trajectory_ids) != 1:
            raise ValueError("adaptive metadata gate requires one trajectory per payload")
        chunks_by_trajectory[trajectory_ids[0]] = chunks
        items.append({
            "group_id": str(payload["group_id"]), "rollout_epoch": epoch_index,
            "trajectory_ids": trajectory_ids, "reward": float(row["reward"]),
            "success": row.get("success"), "steps": int(row.get("steps", 0)),
        })
    filtered, dropped = filter_homogeneous_groups(items, retain_all_success=True)
    advantages = build_smoke_advantages(
        filtered, success_decay_gamma=0.998, action_chunk_steps=16)
    trainable = sum(chunks_by_trajectory[trajectory_id]
                    for trajectory_id, advantage in advantages.items()
                    if float(advantage) != 0.0)
    return {"trainable_action_chunks": trainable,
            "dropped_homogeneous_groups": dropped}


def collect_batch(batch: dict) -> dict:
    """Collect only as many fresh waves as needed under one actor snapshot."""
    expected = batch["policy_snapshot"]
    target = int(batch.get("target_trainable_action_chunks", 1024))
    total = 0
    epoch_index = 0
    while total < target:
        if epoch_index >= 8:
            raise RuntimeError(f"adaptive rollout cap reached with {total}/{target} chunks")
        epoch = {"index": epoch_index,
                 "run_id": f"{batch['batch_id']}-e{epoch_index}", "status": "planned"}
        done = legacy.ssh(
            legacy.SPARK,
            f"test -f {legacy.SPARK_ROOT}/results/{epoch['run_id']}/DONE && echo yes || true",
            capture=True).stdout.strip() == "yes"
        if not done:
            actor = snapshot_from_metrics(legacy.rpc(
                legacy.SPARK, legacy.ACTOR_CONTAINER, {"op": "metrics"}))
            if not same_snapshot(actor, expected):
                raise RuntimeError(
                    f"actor changed before adaptive batch completed: {actor} != {expected}")
        seed_base = int(os.environ.get("RLINF_TRAIN_SEED_BASE", "820000"))
        rows = legacy.collect(
            epoch["run_id"], seed_base + int(batch["attempt"]) * 1000 + int(epoch["index"]) * 100,
            groups=8, group_size=8, workers=4)
        validate_rows_snapshot(rows, expected)
        estimate = estimate_trainable_chunks(rows, epoch_index)
        epoch.update({"status": "ready", "trajectories": len(rows),
                      "successes": sum(bool(row.get("success")) for row in rows),
                      **estimate})
        if epoch_index < len(batch["rollout_epochs"]):
            batch["rollout_epochs"][epoch_index] = epoch
        else:
            batch["rollout_epochs"].append(epoch)
        total += int(estimate["trainable_action_chunks"])
        epoch_index += 1
    batch["rollout_epochs"] = batch["rollout_epochs"][:epoch_index]
    actor_after = snapshot_from_metrics(legacy.rpc(
        legacy.SPARK, legacy.ACTOR_CONTAINER, {"op": "metrics"}))
    if not same_snapshot(actor_after, expected):
        raise RuntimeError("actor adapter changed while an adaptive rollout lease was active")
    batch["trainable_action_chunks_estimate"] = total
    batch["status"] = "ready"
    validate_batch(batch)
    return batch


def consume_streamed(batch: dict, *, recover: bool = False,
                     checkpoint_roundtrip: bool = False) -> dict:
    validate_batch(batch)
    batch_id = batch["batch_id"]
    if recover:
        targets = [f"/results/{batch_id}"] + [
            f"/results/{epoch['run_id']}" for epoch in batch["rollout_epochs"]]
        legacy.ssh(legacy.LEARNER,
                   f"docker run --rm -v {legacy.LEARNER_ROOT}/results:/results "
                   f"alpine rm -rf {' '.join(targets)}")
    snap = batch["policy_snapshot"]
    _remote_stream("begin", batch_id, "--policy-version", str(snap["version"]),
                   "--policy-hash", str(snap["hash"]))
    pipeline_started = time.time()
    epoch_timings = []
    epochs = list(batch["rollout_epochs"])
    # Keep at most one epoch prefetched.  Import/hash validation is CPU+disk bound and
    # previously left the DAX-to-learner link idle for ~30-45 s per epoch.  Starting the
    # next transfer immediately before staging the current epoch overlaps those phases
    # without changing import order, validation, or the durable recovery source.
    with ThreadPoolExecutor(max_workers=1) as transfer_pool:
        transfer_future = transfer_pool.submit(legacy.transfer_run, epochs[0]["run_id"])
        for position, epoch in enumerate(epochs):
            transfer = transfer_future.result()
            if position + 1 < len(epochs):
                next_epoch = epochs[position + 1]
                transfer_future = transfer_pool.submit(
                    legacy.transfer_run, next_epoch["run_id"])
            stage_started = time.time()
            staged = _remote_stream(
                "stage", batch_id, "--run-dir", f"/results/{epoch['run_id']}",
                "--epoch", str(epoch["index"]))
            epoch_timings.append({
                "epoch": int(epoch["index"]),
                "transfer_seconds": float(transfer["elapsed_seconds"]),
                "uncompressed_payload_bytes": int(transfer["uncompressed_payload_bytes"]),
                "groups_transferred": int(transfer["groups_transferred"]),
                "groups_total": int(transfer["groups_total"]),
                "import_verify_seconds": time.time() - stage_started,
                "trainable_action_chunks_after": int(staged["trainable_action_chunks"]),
            })
            # Local transfer scratch is disposable; DAX remains the durable recovery source.
            local_dir = legacy.LOCAL_STAGE / epoch["run_id"]
            if local_dir.exists():
                shutil.rmtree(local_dir)
            archive = legacy.LOCAL_STAGE / f"{epoch['run_id']}.tar"
            if archive.exists():
                archive.unlink()
    finalize_started = time.time()
    extra = ["--checkpoint-roundtrip"] if checkpoint_roundtrip else []
    report = _remote_stream("finalize", batch_id, *extra)
    report["pipeline_timing"] = {
        "epochs": epoch_timings,
        "transfer_import_seconds": finalize_started - pipeline_started,
        "finalize_update_checkpoint_seconds": time.time() - finalize_started,
        "consumer_total_seconds": time.time() - pipeline_started,
    }
    return report


def sync_and_verify(batch: dict, report: dict, learner_before: dict) -> dict:
    next_snapshot = validate_update_report(report, batch, learner_before)
    published = legacy.sync_adapter(batch["batch_id"], next_snapshot["version"])
    actual = {"version": int(published["policy_version"]),
              "hash": str(published["policy_hash"])}
    if not same_snapshot(actual, next_snapshot):
        raise RuntimeError(f"published adapter mismatch: {actual} != {next_snapshot}")
    actor = snapshot_from_metrics(legacy.rpc(
        legacy.SPARK, legacy.ACTOR_CONTAINER, {"op": "metrics"}))
    learner = snapshot_from_metrics(legacy.rpc(
        legacy.LEARNER, legacy.LEARNER_CONTAINER, {"op": "metrics"}))
    if not same_snapshot(actor, next_snapshot) or not same_snapshot(learner, next_snapshot):
        raise RuntimeError("actor/learner did not reach the committed snapshot boundary")
    return next_snapshot


def prune_committed(batch: dict) -> None:
    for epoch in batch["rollout_epochs"]:
        # Re-read rows only after the checkpoint and actor synchronization are durable.
        rows = legacy.collect(epoch["run_id"], 0, groups=8, group_size=8, workers=4)
        legacy.prune_actor_payloads(epoch["run_id"], rows)


def learner_prune_paths(state: dict, *, keep_every: int = 5) -> list[str]:
    """Select committed learner artifacts that no longer need disk residency."""
    if keep_every <= 0:
        raise ValueError("keep_every must be positive")
    history = list(state.get("history", []))
    if not history:
        return []
    latest = max(int(item["snapshot"]["version"]) for item in history)
    paths = []
    for item in history:
        batch_id = str(item["batch_id"])
        if not legacy.SAFE_NAME.fullmatch(batch_id):
            raise ValueError(f"unsafe committed batch ID: {batch_id!r}")
        version = int(item["snapshot"]["version"])
        root = f"/results/{batch_id}/consume_smoke"
        # The actor has already loaded and hash-verified this transfer artifact.
        paths.append(f"{root}/adapter-v{version}.pt")
        # Always retain the newest restart point plus periodic long-term checkpoints.
        if version != latest and version % keep_every != 0:
            paths.append(f"{root}/checkpoint.pt")
    return paths


def prune_learner_artifacts(state: dict, *, keep_every: int = 5) -> None:
    paths = learner_prune_paths(state, keep_every=keep_every)
    if paths:
        legacy.ssh(
            legacy.LEARNER,
            f"docker exec {legacy.LEARNER_CONTAINER} rm -f {' '.join(paths)}")
    for item in state.get("history", []):
        local_adapter = legacy.LOCAL_STAGE / f"adapter-v{int(item['snapshot']['version'])}.pt"
        if local_adapter.exists():
            local_adapter.unlink()


def validate_production_servers(actor_metrics: dict, learner_metrics: dict,
                                *, require_equal: bool = True) -> None:
    """Fail before collection unless both resident servers match the Z-1 contract."""
    if require_equal and not same_snapshot(snapshot_from_metrics(actor_metrics),
                                           snapshot_from_metrics(learner_metrics)):
        raise RuntimeError("actor and learner must start from the exact same policy snapshot")
    config = legacy.rpc(legacy.LEARNER, legacy.LEARNER_CONTAINER,
                        {"op": "config", "code_rev": "z1-async"})["config"]
    expected = {
        "sampler": "pirl", "eta": 0.1,
        "lr": float(os.environ.get("RLINF_EXPECTED_LR", "5e-6")),
        "weight_decay": 0.01, "grad_clip": 1.0,
        "clip": 0.2, "kl_coef": 0.0, "update_epochs": 1,
        "lora_targets": "all_linear", "rank": 16, "alpha": 32,
    }
    mismatches = {
        key: {"actual": config.get(key), "expected": value}
        for key, value in expected.items() if config.get(key) != value
    }
    history = learner_metrics.get("policy_hash_history", {})
    current_version = int(learner_metrics["policy_version"])
    known_hash = history.get(str(current_version), history.get(current_version))
    if known_hash != learner_metrics["policy_hash"]:
        mismatches["policy_hash_history"] = {
            "actual": history, "expected": "current version/hash retained"}
    if learner_metrics.get("fail_stopped"):
        mismatches["fail_stopped"] = {
            "actual": learner_metrics.get("fatal_error"), "expected": False}
    if mismatches:
        raise RuntimeError(f"learner is not async Z-1 ready: {mismatches}")


def validate_fast_hpo_gate(report: dict, *, require_roundtrip: bool) -> dict:
    update = report["update"]
    ratio = float(update.get("epoch0_mean_ratio", float("nan")))
    checks = {
        "epoch0_ratio_approximately_one": math.isfinite(ratio) and abs(ratio - 1.0) <= 0.02,
        "adapter_delta_positive": float(update.get("adapter_delta_l2", 0.0)) > 0.0,
        "nonfinite_zero": int(update.get("n_nonfinite", 0)) +
                          int(update.get("post_step_n_nonfinite", 0)) == 0,
        "dropped_zero": int(update.get("n_dropped", 0)) +
                        int(update.get("post_step_n_dropped", 0)) == 0,
        "clamped_zero": int(update.get("n_clamped", 0)) +
                        int(update.get("post_step_n_clamped", 0)) == 0,
        "post_step_ess": float(update.get("post_step_ess", 0.0)) >= 0.95,
        "post_step_kl": float(update.get("post_step_mean_kl", float("inf"))) <= 0.02,
        "post_step_clip_fraction": float(
            update.get("post_step_clip_fraction", float("inf"))) <= 0.30,
    }
    if require_roundtrip:
        checks["checkpoint_roundtrip"] = bool(
            report.get("checkpoint_roundtrip", {}).get("overall_pass", False))
    result = {"pass": all(checks.values()), "checks": checks}
    if not result["pass"]:
        raise RuntimeError(f"fast HPO update gate failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--state", type=Path,
                        default=legacy.LOCAL_STAGE / "async_state.json")
    parser.add_argument("--run-prefix", default=os.environ.get("RLINF_RUN_PREFIX", "grasp-z1"))
    args = parser.parse_args()
    store = StateStore(args.state)
    with store.locked():
        actor_metrics = legacy.rpc(legacy.SPARK, legacy.ACTOR_CONTAINER, {"op": "metrics"})
        learner_metrics = legacy.rpc(legacy.LEARNER, legacy.LEARNER_CONTAINER, {"op": "metrics"})
        if store.path.exists():
            state = store.load()
            if int(state["target_updates"]) != args.updates:
                raise RuntimeError("cannot change --updates for an existing coordinator state")
        else:
            state = new_state(actor_metrics, learner_metrics, args.updates)
            store.save(state)
        # A durable report may legitimately exist in the narrow window after the
        # learner advanced and before actor synchronization. Recovery below resolves
        # that state; all non-inflight starts still require exact equality.
        validate_production_servers(
            actor_metrics, learner_metrics, require_equal=not bool(state.get("inflight")))

        # A report is the durable optimizer commit record. Without one, only retry if
        # learner metrics prove the optimizer never advanced.
        if state.get("inflight"):
            current = state["inflight"]["current"]
            report_command = (
                f"test -f {legacy.LEARNER_ROOT}/results/{current['batch_id']}/"
                "consume_smoke/consume_smoke.json && echo yes || true")
            exists = legacy.ssh(legacy.LEARNER, report_command, capture=True).stdout.strip() == "yes"
            action = recovery_action(state, learner_metrics, exists)
            if action == "commit_report":
                text = legacy.ssh(legacy.LEARNER,
                    f"cat {legacy.LEARNER_ROOT}/results/{current['batch_id']}/"
                    "consume_smoke/consume_smoke.json", capture=True).stdout
                report = json.loads(text)
            elif action == "retry_same_update_id":
                report = consume_streamed(
                    current, recover=True,
                    checkpoint_roundtrip=state["accepted_updates"] == 0)
            else:  # pragma: no cover - recovery_action is exhaustive
                raise RuntimeError(action)
            validate_fast_hpo_gate(
                report, require_roundtrip=state["accepted_updates"] == 0)
            next_batch = collect_batch(state["inflight"]["next"])
            committed = sync_and_verify(current, report, state["inflight"]["learner_before"])
            state["accepted_updates"] += 1
            state["history"].append({"batch_id": current["batch_id"],
                                     "snapshot": committed,
                                     "checkpoint": report["checkpoint"],
                                     "pipeline_timing": report.get("pipeline_timing")})
            state.update({"actor": committed, "learner": committed,
                          "ready": next_batch, "inflight": None})
            store.save(state)
            prune_committed(current)
            prune_learner_artifacts(state)

        if state["ready"] is None and state["accepted_updates"] < args.updates:
            attempt = state["next_attempt"]
            state["next_attempt"] += 1
            state["ready"] = collect_batch(make_batch(
                f"{args.run_prefix}-b{attempt:04d}-v{state['actor']['version']:04d}",
                state["actor"], attempt))
            store.save(state)

        while state["accepted_updates"] < args.updates:
            current = state["ready"]
            validate_lag(current, state["learner"])
            attempt = state["next_attempt"]
            state["next_attempt"] += 1
            next_batch = make_batch(
                f"{args.run_prefix}-b{attempt:04d}-v{state['actor']['version']:04d}",
                state["actor"], attempt)
            state["inflight"] = {"current": current, "next": next_batch,
                                 "learner_before": state["learner"]}
            state["ready"] = None
            store.save(state)
            with ThreadPoolExecutor(max_workers=2) as pool:
                update_future = pool.submit(
                    consume_streamed, current,
                    checkpoint_roundtrip=state["accepted_updates"] == 0)
                rollout_future = pool.submit(collect_batch, next_batch)
                report = update_future.result()
                next_batch = rollout_future.result()
            report["fast_hpo_gate"] = validate_fast_hpo_gate(
                report, require_roundtrip=state["accepted_updates"] == 0)
            committed = sync_and_verify(current, report, state["inflight"]["learner_before"])
            state["accepted_updates"] += 1
            state["history"].append({"batch_id": current["batch_id"],
                                     "snapshot": committed,
                                     "checkpoint": report["checkpoint"],
                                     "pipeline_timing": report.get("pipeline_timing")})
            state.update({"actor": committed, "learner": committed,
                          "ready": next_batch, "inflight": None})
            store.save(state)
            prune_committed(current)
            prune_learner_artifacts(state)


if __name__ == "__main__":
    try:
        main()
    except AmbiguousUpdateError as error:
        raise SystemExit(f"AMBIGUOUS UPDATE - manual checkpoint audit required: {error}")
