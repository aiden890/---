from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlinf_grid_runtime import collect_with_rlinf


CFG = {
    "run": {"id": "fake-rlinf", "source_commit": "abc", "checkpoint_hash": "ckpt"},
    "grid": {"sampler.noise_level": [0.1, 0.3], "rollout.episodes": 1},
    "seed": {"base": 10, "stride": 100},
    "runtime": {"workers": 2, "max_in_flight": 2, "retries": 0,
                "fail_fast": "per_config", "node_ranks": [0, 0]},
    "model_server": {"host": "127.0.0.1", "port": 1},
}


class Call:
    def __init__(self, result): self.result = result
    def wait(self): return [self.result]


class Proxy:
    def __init__(self, group, rank): self.group, self.rank = group, rank
    def run_episode(self, job):
        if self.group.fail_once and int(job.get("_attempt", 0)) == 0 and job["seed"] == 10:
            return Call({"worker_error": "RuntimeError: planned failure"})
        payload = self.group.output / f"payload-{job['job_id']}.pkl"
        video = self.group.output / f"video-{job['job_id']}.mp4"
        payload.write_bytes(job["job_id"].encode())
        video.write_bytes(b"video")
        result = {
            "worker_id": f"rlinf-{self.rank}",
            "started_at": 1.0,
            "finished_at": 2.0,
            "steps": 1,
            "reward": float(job["seed"] % 2),
            "randomization": {"env_seed": job["seed"]},
            "timing": {"wall_seconds": 1.0},
            "trainer_payload": {
                "path": str(payload), "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                "trajectory_ids": [job["job_id"]], "optimizer_update_requested": False,
            },
            "artifacts": {"trainer_store": str(payload), "video": str(video)},
        }
        return Call(result)


class Group:
    def __init__(self, output, fail_once=False):
        self.output, self.closed, self.fail_once = output, False, fail_once
    def execute_on(self, rank): return Proxy(self, rank)
    def close_client(self): self.closed = True; return Call(True)
    def _close(self): self.closed = True


class Builder:
    group = None
    fail_once = False
    def __init__(self, cfg): self.cfg = cfg
    def launch(self, **kwargs):
        Builder.group = Group(Path(self.cfg["results_root"]) / self.cfg["run_id"], Builder.fail_once)
        return Builder.group


class Worker:
    @classmethod
    def create_group(cls, cfg): return Builder(cfg)


class ProductionRuntimeTests(unittest.TestCase):
    def test_rlinf_worker_group_path_runs_parallel_wave_and_closes(self):
        scheduler = types.ModuleType("rlinf.scheduler")
        scheduler.Cluster = lambda **kwargs: object()
        scheduler.NodePlacementStrategy = lambda ranks: list(ranks)
        rlinf = types.ModuleType("rlinf")
        rlinf.scheduler = scheduler
        worker_module = types.ModuleType("rlinf_grid_worker")
        worker_module.RoboCasaGridWorker = Worker
        old = {name: sys.modules.get(name) for name in ("rlinf", "rlinf.scheduler", "rlinf_grid_worker")}
        sys.modules.update({"rlinf": rlinf, "rlinf.scheduler": scheduler,
                            "rlinf_grid_worker": worker_module})
        try:
            with tempfile.TemporaryDirectory() as td:
                output = Path(td) / "run"
                result = collect_with_rlinf(copy.deepcopy(CFG), output)
                self.assertEqual(result["status"], "done")
                self.assertTrue(result["audit"]["pass"], result)
                self.assertEqual(result["audit"]["max_concurrency"], 2)
                self.assertTrue((output / "audit.json").is_file())
                self.assertTrue(Builder.group.closed)
                rows = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
                self.assertEqual({row["worker_id"] for row in rows}, {"rlinf-0", "rlinf-1"})
        finally:
            for name, module in old.items():
                if module is None: sys.modules.pop(name, None)
                else: sys.modules[name] = module

    def test_worker_error_envelope_is_retried_without_actor_failure(self):
        scheduler = types.ModuleType("rlinf.scheduler")
        scheduler.Cluster = lambda **kwargs: object()
        scheduler.NodePlacementStrategy = lambda ranks: list(ranks)
        rlinf = types.ModuleType("rlinf")
        rlinf.scheduler = scheduler
        worker_module = types.ModuleType("rlinf_grid_worker")
        worker_module.RoboCasaGridWorker = Worker
        old = {name: sys.modules.get(name) for name in ("rlinf", "rlinf.scheduler", "rlinf_grid_worker")}
        sys.modules.update({"rlinf": rlinf, "rlinf.scheduler": scheduler,
                            "rlinf_grid_worker": worker_module})
        Builder.fail_once = True
        try:
            with tempfile.TemporaryDirectory() as td:
                cfg = copy.deepcopy(CFG)
                cfg["runtime"]["retries"] = 1
                output = Path(td) / "run"
                result = collect_with_rlinf(cfg, output)
                self.assertEqual(result["status"], "done")
                self.assertEqual(result["retry_count"], 1)
                failures = [json.loads(line) for line in
                            (output / "failures.jsonl").read_text().splitlines()]
                self.assertEqual(len(failures), 1)
        finally:
            Builder.fail_once = False
            for name, module in old.items():
                if module is None: sys.modules.pop(name, None)
                else: sys.modules[name] = module


if __name__ == "__main__":
    unittest.main()
