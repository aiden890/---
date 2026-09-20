"""Production collection driver using RLinf Worker/WorkerGroup scheduling."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from grid_collection import (
    SCHEMA, CollectionPlan, _RunLock, _append_jsonl, _atomic_json, _read_jsonl,
    audit_collection, validated_completed_ids,
)


def collect_with_rlinf(cfg: dict, output_dir: Path) -> dict:
    """Run simulator episodes through RLinf CPU workers; no local process pool is used."""
    from rlinf.scheduler import Cluster, NodePlacementStrategy
    from rlinf_grid_worker import RoboCasaGridWorker

    plan = CollectionPlan.from_config(cfg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = plan.manifest()
    manifest["runtime_backend"] = "rlinf-worker-group"
    manifest["rlinf_commit"] = "bde6c918642abf9a4776cb1d5fabcc5087dfe195"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("plan_hash") != manifest["plan_hash"]:
            raise RuntimeError("result directory plan hash mismatch")
    else:
        _atomic_json(manifest_path, manifest)

    runtime = cfg.get("runtime", {})
    node_ranks = list(runtime.get("node_ranks", [0] * plan.workers))
    if len(node_ranks) != plan.workers:
        raise ValueError("runtime.node_ranks length must equal runtime.workers")
    cluster = Cluster(num_nodes=max(node_ranks) + 1)
    placement = NodePlacementStrategy(node_ranks)
    worker_cfg = {
        # The output directory is authoritative when run.sh overrides the base YAML run ID.
        "run_id": output_dir.name,
        "results_root": str(output_dir.parent),
        "model_server": dict(cfg.get("model_server", {})),
        "validation": dict(cfg.get("validation", {})),
        "runtime": dict(runtime),
    }
    with _RunLock(output_dir / ".collector.lock", worker_cfg["run_id"]):
        group = RoboCasaGridWorker.create_group(worker_cfg).launch(
            cluster=cluster,
            name=f"MiBoTGrid-{worker_cfg['run_id']}",
            placement_strategy=placement,
            isolate_gpu=False,
            max_concurrency=1,
        )
        (output_dir / "DONE").unlink(missing_ok=True)
        (output_dir / "FAILED").unlink(missing_ok=True)
        episodes_path = output_dir / "episodes.jsonl"
        failures_path = output_dir / "failures.jsonl"
        done_ids = validated_completed_ids(plan, _read_jsonl(episodes_path))
        pending_jobs = [job for job in plan.jobs() if job["job_id"] not in done_ids]
        pending_jobs.sort(key=lambda job: (job["episode_index"], job["config_index"]))
        pending = []
        units: dict[str, list[dict]] = {}
        for job in pending_jobs:
            unit_id = str(job.get("group_id", job["job_id"]))
            if unit_id not in units:
                units[unit_id] = []
                pending.append(units[unit_id])
            units[unit_id].append(job)
        retry_count = 0
        fatal = []
        cancelled_configs: set[str] = set()
        io_lock = __import__("threading").Lock()

        try:
            while pending:
                wave = []
                wave_limit = min(plan.workers, plan.max_in_flight)
                while pending and len(wave) < wave_limit:
                    unit = pending.pop(0)
                    if unit[0]["config_id"] not in cancelled_configs:
                        wave.append(unit)
                calls = []
                for rank, unit in enumerate(wave):
                    proxy = group.execute_on(rank)
                    calls.append(proxy.run_group(unit) if "group_id" in unit[0]
                                 else proxy.run_episode(unit[0]))
                for rank, (unit, call) in enumerate(zip(wave, calls)):
                    first = unit[0]
                    try:
                        envelope = call.wait()[0]
                        if envelope.get("worker_error"):
                            raise RuntimeError(envelope["worker_error"])
                        results = envelope["results"] if "group_id" in first else [envelope]
                        result_by_id = {result.get("job_id", first["job_id"]): result
                                        for result in results}
                        if set(result_by_id) != {job["job_id"] for job in unit}:
                            raise RuntimeError("worker returned incomplete or mixed group results")
                        records = []
                        for job in unit:
                            record = {"schema": SCHEMA, **job, **result_by_id[job["job_id"]]}
                            if record["trainer_payload"].get(
                                    "optimizer_update_requested") is not False:
                                raise RuntimeError(
                                    "worker requested optimizer update during collection")
                            records.append(record)
                        for record in records:
                            _append_jsonl(episodes_path, record, io_lock)
                            config_dir = output_dir / record["config_id"]
                            config_dir.mkdir(exist_ok=True)
                            _append_jsonl(config_dir / "episodes.jsonl", record, io_lock)
                    except Exception as exc:
                        attempts = int(first.get("_attempt", 0)) + 1
                        failure = {**first, "worker_id": f"rlinf-{rank}",
                                   "attempt": attempts, "group_jobs": len(unit),
                                   "error": f"{type(exc).__name__}: {exc}", "ts": time.time()}
                        _append_jsonl(failures_path, failure, io_lock)
                        if attempts <= plan.retries:
                            retry_count += 1
                            for job in unit:
                                job["_attempt"] = attempts
                            pending.append(unit)
                        else:
                            fatal.append(failure)
                            if plan.fail_fast == "global":
                                pending.clear()
                            elif plan.fail_fast == "per_config":
                                cancelled_configs.add(first["config_id"])
                                pending = [item for item in pending
                                           if item[0]["config_id"] != first["config_id"]]
        finally:
            try:
                group.close_client().wait()
            finally:
                group._close()

        rows = _read_jsonl(episodes_path)
        expected = {job["job_id"] for job in plan.jobs()}
        completed = {row["job_id"] for row in rows}
        audit = audit_collection(output_dir)
        _atomic_json(output_dir / "audit.json", audit)
        status = "done" if not fatal and completed == expected and audit["pass"] else "failed"
        summary = {
            "schema": SCHEMA,
            "status": status,
            "planned_episodes": len(expected),
            "completed_episodes": len(completed & expected),
            "retry_count": retry_count,
            "fatal_failures": fatal,
            "optimizer_update_requests": 0,
            "audit": audit,
            "pid": os.getpid(),
        }
        _atomic_json(output_dir / "summary.json", summary)
        _atomic_json(output_dir / ("DONE" if status == "done" else "FAILED"), summary)
        return summary
