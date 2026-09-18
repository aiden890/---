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
        pending = [job for job in plan.jobs() if job["job_id"] not in done_ids]
        pending.sort(key=lambda job: (job["episode_index"], job["config_index"]))
        retry_count = 0
        fatal = []
        cancelled_configs: set[str] = set()
        io_lock = __import__("threading").Lock()

        try:
            while pending:
                wave = []
                wave_limit = min(plan.workers, plan.max_in_flight)
                while pending and len(wave) < wave_limit:
                    job = pending.pop(0)
                    if job["config_id"] not in cancelled_configs:
                        wave.append(job)
                calls = [group.execute_on(rank).run_episode(job)
                         for rank, job in enumerate(wave)]
                for rank, (job, call) in enumerate(zip(wave, calls)):
                    try:
                        result = call.wait()[0]
                        record = {"schema": SCHEMA, **job, **result}
                        if record["trainer_payload"].get("optimizer_update_requested") is not False:
                            raise RuntimeError("worker requested optimizer update during collection")
                        _append_jsonl(episodes_path, record, io_lock)
                        config_dir = output_dir / job["config_id"]
                        config_dir.mkdir(exist_ok=True)
                        _append_jsonl(config_dir / "episodes.jsonl", record, io_lock)
                    except Exception as exc:
                        attempts = int(job.get("_attempt", 0)) + 1
                        failure = {**job, "worker_id": f"rlinf-{rank}", "attempt": attempts,
                                   "error": f"{type(exc).__name__}: {exc}", "ts": time.time()}
                        _append_jsonl(failures_path, failure, io_lock)
                        if attempts <= plan.retries:
                            retry_count += 1
                            job["_attempt"] = attempts
                            pending.append(job)
                        else:
                            fatal.append(failure)
                            if plan.fail_fast == "global":
                                pending.clear()
                            elif plan.fail_fast == "per_config":
                                cancelled_configs.add(job["config_id"])
                                pending = [item for item in pending
                                           if item["config_id"] != job["config_id"]]
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
