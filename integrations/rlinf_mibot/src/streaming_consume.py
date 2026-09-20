#!/usr/bin/env python3
"""Disk-bounded four-epoch collector consumer.

Each epoch is validated and imported into the trainer's CPU rollout store before its
multi-GB learner-side files are removed.  The final call performs exactly one logical
optimizer update and saves/reloads a checkpoint.  Actor-side sources are intentionally
outside this module and remain available until the coordinator commits the update.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from consume_collector import (
    TrainerRPC,
    build_smoke_advantages,
    filter_homogeneous_groups,
    inspect_action_chunks,
    load_collector_payloads,
    optimizer_update_epochs,
    require_trainable_action_chunks,
    validate_rollout_batch,
    validate_update,
)


SCHEMA = "mibot-streamed-consume-v1"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def begin(batch_dir: Path, host: str, port: int, policy_snapshot: dict[str, Any]) -> dict[str, Any]:
    batch_dir = Path(batch_dir)
    manifest_path = batch_dir / "streaming_manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"streaming manifest already exists: {manifest_path}")
    client = TrainerRPC(host, port)
    try:
        reset = client.call({"op": "reset"})
    finally:
        client.close()
    manifest = {
        "schema": SCHEMA, "status": "staging", "policy_snapshot": policy_snapshot,
        "rollout_epochs": [], "compact_payloads": [], "advantages": {},
        "trainable_action_chunks": 0, "action_chunks_total": 0, "reset": reset,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def stage_epoch(batch_dir: Path, run_dir: Path, epoch_index: int,
                host: str, port: int, cfg: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(batch_dir) / "streaming_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "staging":
        raise RuntimeError("streamed batch is not in staging state")
    epoch_index = int(epoch_index)
    if epoch_index != len(manifest["rollout_epochs"]):
        raise ValueError("rollout epochs must be staged once and in order")
    payloads = load_collector_payloads(Path(run_dir))
    for item in payloads:
        item["rollout_epoch"] = epoch_index
    validate_rollout_batch(payloads, expected_group_size=int(cfg["expected_group_size"]))
    identities = {(item["policy_version"], item["policy_hash"]) for item in payloads}
    wanted = {(int(manifest["policy_snapshot"]["version"]),
               str(manifest["policy_snapshot"]["hash"]))}
    if identities != wanted:
        raise ValueError(f"streamed epoch policy mismatch: {identities} != {wanted}")
    filtered, dropped = filter_homogeneous_groups(
        payloads, retain_all_success=cfg.get("success_decay_gamma") is not None)
    if not filtered:
        raise ValueError("rollout epoch has no trainable groups after filtering")
    advantages = build_smoke_advantages(
        filtered, success_decay_gamma=cfg.get("success_decay_gamma"),
        action_chunk_steps=int(cfg.get("action_chunk_steps", 16)))
    chunks = inspect_action_chunks(filtered, advantages)
    client = TrainerRPC(host, port)
    imports = []
    try:
        for item in filtered:
            imports.append(client.call({
                "op": "import_store", "path": item["path"],
                "expected_sha256": item["sha256"], "max_policy_lag": 1,
            }))
    finally:
        client.close()
    # Delete only after every import succeeded. A partial failure preserves files for
    # diagnosis; recovery resets the trainer store and restages from DAX sources.
    for item in payloads:
        Path(item["path"]).unlink()
    compact = [{key: item[key] for key in (
        "group_id", "trajectory_ids", "rollout_epoch", "policy_version", "policy_hash")}
        for item in filtered]
    duplicate = set(manifest["advantages"]).intersection(advantages)
    if duplicate:
        raise ValueError(f"duplicate trajectory ids across rollout epochs: {sorted(duplicate)}")
    manifest["rollout_epochs"].append({
        "index": epoch_index, "run_dir": str(run_dir), "imports": len(imports),
        "dropped_homogeneous_groups": dropped,
        "trainable_action_chunks": chunks["trainable_action_chunks"],
    })
    manifest["compact_payloads"].extend(compact)
    manifest["advantages"].update(advantages)
    manifest["trainable_action_chunks"] += int(chunks["trainable_action_chunks"])
    manifest["action_chunks_total"] += int(chunks["action_chunks_total"])
    _atomic_json(manifest_path, manifest)
    return manifest


def finalize(batch_dir: Path, host: str, port: int, cfg: dict[str, Any]) -> dict[str, Any]:
    batch_dir = Path(batch_dir)
    manifest_path = batch_dir / "streaming_manifest.json"
    report_path = batch_dir / "consume_smoke" / "consume_smoke.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    collection = validate_rollout_batch(
        manifest["compact_payloads"], expected_group_size=int(cfg["expected_group_size"]),
        min_rollout_epochs=int(cfg["min_rollout_epochs"]))
    require_trainable_action_chunks(
        {"trainable_action_chunks": manifest["trainable_action_chunks"]},
        int(cfg["min_trainable_action_chunks"]))
    if optimizer_update_epochs(cfg) != 1:
        raise ValueError("streamed Z-1 production updates require optimizer_update_epochs=1")
    checkpoint = batch_dir / "consume_smoke" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    client = TrainerRPC(host, port)
    started = time.time()
    timings: dict[str, float] = {}
    try:
        phase_started = time.time()
        before = client.call({"op": "metrics"})
        timings["metrics_seconds"] = time.time() - phase_started
        phase_started = time.time()
        update = client.call({
            "op": "update", "advantages": manifest["advantages"],
            "clip": float(cfg.get("clip", 0.2)),
            "kl_coef": float(cfg.get("kl_coef", 0.0)),
            "ratio_max": float(cfg.get("ratio_max", 10.0)),
            "adv_clip": float(cfg.get("adv_clip", 3.0)),
            "update_epochs": 1,
            "target_kl": cfg.get("target_kl"),
            "update_id": f"collector-consume-{batch_dir.name}",
            "curriculum_state": None,
        })
        timings["optimizer_update_seconds"] = time.time() - phase_started
        phase_started = time.time()
        saved = client.call({
            "op": "save", "path": str(checkpoint),
            "update_index": int(update["policy_version_after"]),
            "train_meta": {"source": "rlinf_streaming_consume",
                           "rollout_epochs": collection["rollout_epoch_ids"]},
        })
        timings["checkpoint_save_seconds"] = time.time() - phase_started
        phase_started = time.time()
        loaded = client.call({"op": "load", "path": str(checkpoint)})
        timings["checkpoint_reload_seconds"] = time.time() - phase_started
    finally:
        client.close()
    gate = validate_update(update)
    checkpoint_ok = checkpoint.is_file() and bool(saved) and bool(loaded)
    report = {
        "status": "PASS" if gate["pass"] and checkpoint_ok else "FAIL",
        "run_dir": str(batch_dir), "checkpoint": str(checkpoint),
        "checkpoint_reloadable": checkpoint_ok, "store_before_update": before,
        "update": update, "update_gate": gate, "saved": saved, "loaded": loaded,
        "algorithm_diagnostics": {
            "collection": collection,
            "trainable_action_chunks": manifest["trainable_action_chunks"],
            "action_chunks_total": manifest["action_chunks_total"],
            "optimizer_update_epochs": 1,
        },
        "stage_timings": timings,
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(report_path, report)
    manifest["status"] = "updated" if report["status"] == "PASS" else "failed"
    _atomic_json(manifest_path, manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("begin", "stage", "finalize"))
    parser.add_argument("batch_dir")
    parser.add_argument("--run-dir")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10088)
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy-version", type=int)
    parser.add_argument("--policy-hash")
    args = parser.parse_args()
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())["consume_smoke"]
    if args.mode == "begin":
        result = begin(Path(args.batch_dir), args.host, args.port,
                       {"version": args.policy_version, "hash": args.policy_hash})
    elif args.mode == "stage":
        result = stage_epoch(Path(args.batch_dir), Path(args.run_dir), args.epoch,
                             args.host, args.port, cfg)
    else:
        result = finalize(Path(args.batch_dir), args.host, args.port, cfg)
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
