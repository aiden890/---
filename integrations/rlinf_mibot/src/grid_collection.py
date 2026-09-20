"""Deterministic parameter-grid rollout collection.

The collector is deliberately optimizer-free. Production execution delegates episodes to
RLinf workers (see ``rlinf_grid_runtime.py``); the in-process fake backend exists only for
CPU scheduling, crash, resume, and artifact-contract tests.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

SCHEMA = "mibot-grpo-rollout-v1"
SUPPORTED_GRID_KEYS = {
    "sampler.noise_level", "sampler.name", "env.skill", "env.split", "env.horizon",
    "env.horizon_grasp", "env.horizon_move", "env.horizon_place", "rollout.episodes",
    "rollout.groups", "rollout.group_size",
    "rollout.replan_steps", "rollout.obs_history", "rollout.obs_interval",
    "rollout.save_video",
    "rollout.video_stride", "rollout.video_fps", "reward.variant",
    "reward.use_milestones", "reward.skill_success", "reward.success_gamma",
    "reward.success_decay",
}


def jsonable(value: Any) -> Any:
    """Recursively normalize numpy/simulator values into portable JSON values."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "item"):
        return jsonable(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def expand_parameter_grid(grid: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand sorted dotted-key values into stable Cartesian-product configurations."""
    if not grid:
        raise ValueError("grid must contain at least one parameter")
    keys = sorted(grid)
    unsupported = sorted(set(keys) - SUPPORTED_GRID_KEYS)
    if unsupported:
        raise ValueError(f"unsupported grid parameters: {unsupported}")
    choices = []
    for key in keys:
        value = grid[key]
        values = list(value) if isinstance(value, list) else [value]
        if not values:
            raise ValueError(f"grid parameter {key!r} has no values")
        choices.append(values)

    expanded = []
    for values in itertools.product(*choices):
        parameters = dict(zip(keys, values))
        config_hash = _hash(parameters)
        expanded.append({
            "config_id": f"cfg-{config_hash[:12]}",
            "config_hash": config_hash,
            "parameters": parameters,
        })
    hashes = [item["config_hash"] for item in expanded]
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate expanded parameter configurations are not allowed")
    return expanded


@dataclass(frozen=True)
class CollectionPlan:
    run: dict[str, Any]
    configs: list[dict[str, Any]]
    seed_base: int
    seed_stride: int
    workers: int
    max_in_flight: int
    retries: int
    fail_fast: str

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "CollectionPlan":
        configs = expand_parameter_grid(dict(cfg["grid"]))
        unsupported_samplers = sorted({str(item["parameters"].get("sampler.name"))
                                       for item in configs
                                       if item["parameters"].get("sampler.name", "pirl_flow_sde") !=
                                       "pirl_flow_sde"})
        if unsupported_samplers:
            raise ValueError(f"launcher currently supports sampler.name=pirl_flow_sde only: "
                             f"{unsupported_samplers}")
        seed_cfg = cfg.get("seed", {})
        runtime = cfg.get("runtime", {})
        episodes = {int(c["parameters"].get("rollout.episodes", 1)) for c in configs}
        if any(n < 1 for n in episodes):
            raise ValueError("rollout.episodes must be >= 1")
        group_counts = []
        for config in configs:
            parameters = config["parameters"]
            has_groups = "rollout.groups" in parameters
            has_group_size = "rollout.group_size" in parameters
            if has_groups != has_group_size:
                raise ValueError("rollout.groups and rollout.group_size must be set together")
            if has_groups:
                groups = int(parameters["rollout.groups"])
                group_size = int(parameters["rollout.group_size"])
                if groups < 1 or group_size < 1:
                    raise ValueError("rollout.groups and rollout.group_size must be >= 1")
                group_counts.append(groups)
        stride = int(seed_cfg.get("stride", 100000))
        required_stride = max([*episodes, *group_counts])
        if stride < required_stride:
            raise ValueError("seed.stride must cover the largest episode or group count")
        workers = int(runtime.get("workers", 1))
        max_in_flight = int(runtime.get("max_in_flight", workers))
        if workers < 1 or max_in_flight < 1:
            raise ValueError("runtime workers/max_in_flight must be >= 1")
        fail_fast = str(runtime.get("fail_fast", "per_config"))
        if fail_fast not in {"per_config", "global", "never"}:
            raise ValueError("runtime.fail_fast must be per_config, global, or never")
        retries = int(runtime.get("retries", 0))
        if retries < 0:
            raise ValueError("runtime.retries must be >= 0")
        return cls(
            run=copy.deepcopy(cfg.get("run", {})),
            configs=configs,
            seed_base=int(seed_cfg.get("base", 0)),
            seed_stride=stride,
            workers=workers,
            max_in_flight=max_in_flight,
            retries=retries,
            fail_fast=fail_fast,
        )

    def jobs(self) -> list[dict[str, Any]]:
        jobs = []
        for config_index, config in enumerate(self.configs):
            parameters = config["parameters"]
            if "rollout.groups" in parameters:
                group_count = int(parameters["rollout.groups"])
                group_size = int(parameters["rollout.group_size"])
                for group_index in range(group_count):
                    seed = self.seed_base + config_index * self.seed_stride + group_index
                    group_id = f"{config['config_id']}-g{group_index:05d}"
                    for member_index in range(group_size):
                        episode_index = group_index * group_size + member_index
                        jobs.append({
                            "job_id": f"{group_id}-m{member_index:03d}",
                            "group_id": group_id,
                            "group_index": group_index,
                            "member_index": member_index,
                            "action_seed": seed * 100000 + member_index,
                            "config_index": config_index,
                            "config_id": config["config_id"],
                            "config_hash": config["config_hash"],
                            "parameters": copy.deepcopy(parameters),
                            "episode_index": episode_index,
                            "seed": seed,
                        })
            else:
                count = int(parameters.get("rollout.episodes", 1))
                for episode_index in range(count):
                    seed = self.seed_base + config_index * self.seed_stride + episode_index
                    jobs.append({
                        "job_id": f"{config['config_id']}-ep{episode_index:05d}",
                        "config_index": config_index,
                        "config_id": config["config_id"],
                        "config_hash": config["config_hash"],
                        "parameters": copy.deepcopy(parameters),
                        "episode_index": episode_index,
                        "seed": seed,
                    })
        seed_owners: dict[int, str] = {}
        for job in jobs:
            owner = str(job.get("group_id", job["job_id"]))
            old_owner = seed_owners.setdefault(int(job["seed"]), owner)
            if old_owner != owner:
                raise ValueError("seed allocator produced a seed shared across groups")
        return jobs

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "run": self.run,
            "plan_hash": _hash({"run": self.run, "configs": self.configs,
                                "seed_base": self.seed_base, "seed_stride": self.seed_stride}),
            "configs": self.configs,
            "jobs": self.jobs(),
            "seed": {"base": self.seed_base, "stride": self.seed_stride},
            "runtime": {"workers": self.workers, "max_in_flight": self.max_in_flight,
                        "retries": self.retries, "fail_fast": self.fail_fast},
            "optimizer_update_requests": 0,
        }


class EpisodeBackend(Protocol):
    def run_episode(self, job: dict[str, Any], worker_id: str) -> dict[str, Any]: ...


class FakeEpisodeBackend:
    """Deterministic test backend; never used by the production launcher."""

    def __init__(self, delay_seconds: float = 0.0, fail_once_seeds: set[int] | None = None):
        self.delay_seconds = float(delay_seconds)
        self.fail_once_seeds = set(fail_once_seeds or set())
        self._failed: set[int] = set()
        self._lock = threading.Lock()

    def run_episode(self, job: dict[str, Any], worker_id: str) -> dict[str, Any]:
        started = time.time()
        with self._lock:
            should_fail = job["seed"] in self.fail_once_seeds and job["seed"] not in self._failed
            if should_fail:
                self._failed.add(job["seed"])
        if should_fail:
            raise RuntimeError(f"injected worker crash for seed {job['seed']}")
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        noise = float(job["parameters"].get("sampler.noise_level", 0.0))
        success = ((int(job["seed"]) + int(noise * 10)) % 2) == 0
        return {
            "started_at": started,
            "finished_at": time.time(),
            "steps": int(job["parameters"].get("env.horizon", 1)),
            "skill_outcome": "SUCCESS" if success else "TIMEOUT",
            "reward": 1.0 if success else 0.0,
            "randomization": {"fake_seed": int(job["seed"])},
            "timing": {"wall_seconds": max(0.0, time.time() - started)},
            "trainer_payload": {
                "schema": "grpo-trainer-store-ref-v1",
                "trajectory_ids": [job["job_id"]],
                "optimizer_update_requested": False,
            },
        }


class _RunLock:
    def __init__(self, path: Path, run_id: str):
        self.path = path
        self.run_id = run_id
        self.fd: int | None = None

    def __enter__(self):
        payload = _canonical({"pid": os.getpid(), "run_id": self.run_id}) + "\n"
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            raise RuntimeError(f"collector lock already exists: {self.path}") from exc
        os.write(self.fd, payload.encode("utf-8"))
        os.fsync(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _append_jsonl(path: Path, value: Any, lock: threading.Lock) -> None:
    line = json.dumps(value, sort_keys=True) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validated_completed_ids(plan: CollectionPlan, rows: list[dict[str, Any]]) -> set[str]:
    """Validate persisted rows before allowing them to suppress work during resume."""
    expected = {job["job_id"]: job for job in plan.jobs()}
    seen: set[str] = set()
    for row in rows:
        job_id = str(row.get("job_id"))
        if job_id in seen:
            raise ValueError(f"duplicate resume row for job_id {job_id!r}")
        job = expected.get(job_id)
        if job is None:
            raise ValueError(f"unknown resume row job_id {job_id!r}")
        keys = ["config_id", "config_hash", "parameters", "episode_index", "seed"]
        keys.extend(key for key in ("group_id", "group_index", "member_index", "action_seed")
                    if key in expected)
        for key in keys:
            if row.get(key) != job[key]:
                raise ValueError(f"resume row {job_id!r} has mismatched {key}")
        seen.add(job_id)
    return seen


def collect_grid(cfg: dict[str, Any], output_dir: Path, backend: EpisodeBackend,
                 max_new_episodes: int | None = None) -> dict[str, Any]:
    """Collect pending jobs with bounded concurrency, retry, resume, and sentinels."""
    plan = CollectionPlan.from_config(cfg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = plan.manifest()
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("plan_hash") != manifest["plan_hash"]:
            raise RuntimeError("existing result directory belongs to a different collection plan")
    else:
        _atomic_json(manifest_path, manifest)

    with _RunLock(output_dir / ".collector.lock", str(plan.run.get("id", "grid"))):
        (output_dir / "DONE").unlink(missing_ok=True)
        (output_dir / "FAILED").unlink(missing_ok=True)
        episode_path = output_dir / "episodes.jsonl"
        failure_path = output_dir / "failures.jsonl"
        existing_rows = _read_jsonl(episode_path)
        completed = validated_completed_ids(plan, existing_rows)
        pending = [job for job in plan.jobs() if job["job_id"] not in completed]
        # Round-robin configs so the first worker wave proves cross-config concurrency
        # instead of consuming every episode of config 0 before config 1 starts.
        pending.sort(key=lambda job: (job["episode_index"], job["config_index"]))
        if max_new_episodes is not None:
            pending = pending[: max(0, int(max_new_episodes))]
        io_lock = threading.Lock()
        retry_count = 0
        fatal: list[dict[str, Any]] = []
        cancelled_configs: set[str] = set()

        def execute(job: dict[str, Any], worker_index: int) -> dict[str, Any]:
            nonlocal retry_count
            worker_id = f"fake-{worker_index}" if isinstance(backend, FakeEpisodeBackend) else f"worker-{worker_index}"
            last_error = None
            for attempt in range(1, plan.retries + 2):
                try:
                    result = backend.run_episode(copy.deepcopy(job), worker_id)
                    record = {"schema": SCHEMA, **job, "worker_id": worker_id, **result}
                    if record.get("trainer_payload", {}).get("optimizer_update_requested") is not False:
                        raise RuntimeError("collector backend requested an optimizer update")
                    _append_jsonl(episode_path, record, io_lock)
                    config_dir = output_dir / job["config_id"]
                    config_dir.mkdir(exist_ok=True)
                    _append_jsonl(config_dir / "episodes.jsonl", record, io_lock)
                    return record
                except Exception as exc:
                    last_error = exc
                    failure = {**job, "worker_id": worker_id, "attempt": attempt,
                               "error": f"{type(exc).__name__}: {exc}", "ts": time.time()}
                    _append_jsonl(failure_path, failure, io_lock)
                    if attempt <= plan.retries:
                        with io_lock:
                            retry_count += 1
                        continue
                    raise RuntimeError(f"job {job['job_id']} failed after {attempt} attempts: {last_error}")
            raise AssertionError("unreachable")

        max_workers = min(plan.workers, plan.max_in_flight)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="grid-fake") as pool:
            futures = {}
            for index, job in enumerate(pending):
                if job["config_id"] in cancelled_configs:
                    continue
                futures[pool.submit(execute, job, index % max_workers)] = job
            for future in as_completed(futures):
                job = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    fatal.append({"job_id": job["job_id"], "config_id": job["config_id"],
                                  "error": str(exc)})
                    if plan.fail_fast == "per_config":
                        cancelled_configs.add(job["config_id"])
                    elif plan.fail_fast == "global":
                        for other in futures:
                            other.cancel()

        all_rows = _read_jsonl(episode_path)
        expected_ids = {job["job_id"] for job in plan.jobs()}
        done_ids = {row["job_id"] for row in all_rows}
        status = "done" if not fatal and done_ids == expected_ids else ("failed" if fatal else "partial")
        summary = {
            "schema": SCHEMA,
            "status": status,
            "planned_episodes": len(expected_ids),
            "completed_episodes": len(done_ids & expected_ids),
            "retry_count": retry_count,
            "fatal_failures": fatal,
            "optimizer_update_requests": 0,
        }
        _atomic_json(output_dir / "summary.json", summary)
        if status == "done":
            _atomic_json(output_dir / "DONE", summary)
        elif status == "failed":
            _atomic_json(output_dir / "FAILED", summary)
        return summary


def audit_collection(output_dir: Path) -> dict[str, Any]:
    """Fail-closed artifact audit, including proof that configs overlapped in wall time."""
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    rows = _read_jsonl(output_dir / "episodes.jsonl")
    by_id = {item["config_id"]: item for item in manifest["configs"]}
    seeds = [int(row["seed"]) for row in rows]
    job_ids = [row.get("job_id") for row in rows]
    expected_jobs = {job["job_id"]: job for job in manifest.get("jobs", [])}
    seed_groups: dict[int, set[str]] = {}
    for row in rows:
        seed_groups.setdefault(int(row["seed"]), set()).add(
            str(row.get("group_id", row.get("job_id"))))
    duplicate_seed_count = sum(max(0, len(owners) - 1) for owners in seed_groups.values())
    mixing = 0
    job_mismatches = 0
    update_requests = 0
    for row in rows:
        expected_job = expected_jobs.get(row.get("job_id"))
        match_keys = ["config_id", "config_hash", "parameters", "episode_index", "seed"]
        if expected_job is not None:
            match_keys.extend(key for key in
                              ("group_id", "group_index", "member_index", "action_seed")
                              if key in expected_job)
        if expected_job is None or any(
                row.get(key) != expected_job[key] for key in match_keys):
            job_mismatches += 1
        expected = by_id.get(row.get("config_id"))
        if (expected is None or row.get("config_hash") != expected["config_hash"] or
                row.get("parameters") != expected["parameters"]):
            mixing += 1
        update_requests += int(row.get("trainer_payload", {}).get("optimizer_update_requested") is not False)

    events = []
    for row in rows:
        events.append((float(row["started_at"]), 1))
        events.append((float(row["finished_at"]), -1))
    active = 0
    max_concurrency = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        max_concurrency = max(max_concurrency, active)

    overlap = 0.0
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            if left["config_id"] == right["config_id"]:
                continue
            overlap = max(overlap, min(float(left["finished_at"]), float(right["finished_at"])) -
                          max(float(left["started_at"]), float(right["started_at"])))
    overlap = max(0.0, overlap)
    runtime_manifest = manifest.get("runtime", {})
    expected_workers = min(int(runtime_manifest.get("workers", 1)),
                           int(runtime_manifest.get("max_in_flight", 1)))
    parallel_expected = expected_workers >= 2
    checks = {
        "planned_configs_present": bool(by_id) and
        {row.get("config_id") for row in rows} == set(by_id),
        "episodes_present": bool(rows),
        "no_duplicate_seeds": duplicate_seed_count == 0,
        "no_config_mixing": mixing == 0,
        "no_optimizer_updates": update_requests == 0 and manifest.get("optimizer_update_requests") == 0,
        "no_duplicate_job_ids": len(job_ids) == len(set(job_ids)),
        "exact_expected_job_coverage": set(job_ids) == set(expected_jobs),
        "rows_match_planned_jobs": job_mismatches == 0,
    }
    if parallel_expected:
        checks["parallel_workers_observed"] = max_concurrency >= 2
        # Cross-config overlap is a meaningful scheduling gate only when the plan
        # actually contains multiple configurations. A single-config GRPO wave still
        # proves parallelism via max_concurrency, but can never overlap two configs.
        if len(by_id) >= 2:
            checks["cross_config_overlap_observed"] = overlap > 0.0
    else:
        checks["serial_execution_observed"] = max_concurrency <= 1
    if manifest.get("runtime_backend") == "rlinf-worker-group":
        artifacts_present = True
        payload_hashes_valid = True
        root = output_dir.resolve()
        for row in rows:
            artifacts = row.get("artifacts", {})
            payload = Path(artifacts.get("trainer_store", ""))
            video_value = artifacts.get("video")
            video_required = bool(row.get("parameters", {}).get("rollout.save_video", True))
            try:
                payload.resolve().relative_to(root)
                payload_in_root = True
            except ValueError:
                payload_in_root = False
            payload_ok = (payload_in_root and not payload.is_symlink() and payload.is_file())
            video_ok = video_value in (None, "")
            if video_required and video_value:
                video = Path(video_value)
                try:
                    video.resolve().relative_to(root)
                    video_in_root = True
                except ValueError:
                    video_in_root = False
                video_ok = video_in_root and not video.is_symlink() and video.is_file()
            if not payload_ok or not video_ok:
                artifacts_present = False
            expected_hash = row.get("trainer_payload", {}).get("sha256")
            if not payload.is_file() or not expected_hash:
                payload_hashes_valid = False
            elif hashlib.sha256(payload.read_bytes()).hexdigest() != expected_hash:
                payload_hashes_valid = False
        checks["production_artifacts_present"] = artifacts_present
        checks["trainer_payload_hashes_valid"] = payload_hashes_valid
    return {
        "pass": all(checks.values()), "checks": checks,
        "config_count": len(by_id), "episode_count": len(rows),
        "duplicate_seed_count": duplicate_seed_count,
        "config_mixing_count": mixing,
        "job_mismatch_count": job_mismatches,
        "optimizer_update_requests": update_requests,
        "max_concurrency": max_concurrency,
        "cross_config_overlap_seconds": overlap,
    }
