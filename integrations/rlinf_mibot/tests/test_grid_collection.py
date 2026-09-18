from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grid_collection import (  # noqa: E402
    CollectionPlan,
    FakeEpisodeBackend,
    audit_collection,
    collect_grid,
    expand_parameter_grid,
    jsonable,
)
from grid_entry import resolve_provenance  # noqa: E402


BASE_CONFIG = {
    "run": {"id": "unit-grid", "source_commit": "abc123", "checkpoint_hash": "ckpt123"},
    "grid": {
        "sampler.noise_level": [0.1, 0.3],
        "env.skill": "GRASP_OBJECT",
        "env.horizon": 8,
        "rollout.episodes": 2,
    },
    "seed": {"base": 100, "stride": 1000},
    "runtime": {
        "workers": 2,
        "max_in_flight": 2,
        "retries": 1,
        "fail_fast": "per_config",
    },
}


class GridExpansionTests(unittest.TestCase):
    def test_container_provenance_uses_injected_commit(self):
        cfg = json.loads(json.dumps(BASE_CONFIG))
        cfg["run"]["source_commit"] = "auto"
        with patch.dict("os.environ", {"RLINF_SOURCE_COMMIT": "deadbeef"}):
            resolved = resolve_provenance(cfg)
        self.assertEqual(resolved["run"]["source_commit"], "deadbeef")

    def test_duplicate_grid_values_are_rejected(self):
        duplicate = json.loads(json.dumps(BASE_CONFIG))
        duplicate["grid"]["sampler.noise_level"] = [0.1, 0.1]
        with self.assertRaisesRegex(ValueError, "duplicate expanded"):
            CollectionPlan.from_config(duplicate)

    def test_negative_retries_are_rejected(self):
        invalid = json.loads(json.dumps(BASE_CONFIG))
        invalid["runtime"]["retries"] = -1
        with self.assertRaisesRegex(ValueError, "retries"):
            CollectionPlan.from_config(invalid)

    def test_unsupported_grid_key_is_rejected(self):
        invalid = json.loads(json.dumps(BASE_CONFIG))
        invalid["grid"]["ignored.setting"] = 1
        with self.assertRaisesRegex(ValueError, "unsupported grid"):
            CollectionPlan.from_config(invalid)

    def test_sampler_that_launcher_cannot_honor_is_rejected(self):
        invalid = json.loads(json.dumps(BASE_CONFIG))
        invalid["grid"]["sampler.name"] = "fixed_noise"
        with self.assertRaisesRegex(ValueError, "pirl_flow_sde only"):
            CollectionPlan.from_config(invalid)

    def test_jsonable_converts_array_scalars_and_nested_arrays(self):
        import numpy as np
        value = {"flag": np.bool_(True), "pose": np.asarray([1.0, 2.0]),
                 "nested": (np.int64(3),)}
        converted = jsonable(value)
        self.assertEqual(converted, {"flag": True, "pose": [1.0, 2.0], "nested": [3]})
        json.dumps(converted)

    def test_expansion_is_stable_and_cartesian(self):
        reversed_grid = dict(reversed(list(BASE_CONFIG["grid"].items())))
        first = expand_parameter_grid(BASE_CONFIG["grid"])
        second = expand_parameter_grid(reversed_grid)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual([item["parameters"]["sampler.noise_level"] for item in first], [0.1, 0.3])
        self.assertEqual(len({item["config_hash"] for item in first}), 2)
        self.assertTrue(all(item["config_id"].startswith("cfg-") for item in first))

    def test_plan_allocates_disjoint_reproducible_seeds(self):
        plan = CollectionPlan.from_config(BASE_CONFIG)
        jobs = plan.jobs()
        self.assertEqual(len(jobs), 4)
        seeds = [job["seed"] for job in jobs]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(seeds, [100, 101, 1100, 1101])
        self.assertEqual(jobs, CollectionPlan.from_config(BASE_CONFIG).jobs())


class CollectionTests(unittest.TestCase):
    def test_serial_baseline_audits_without_requiring_parallel_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            serial = json.loads(json.dumps(BASE_CONFIG))
            serial["runtime"].update({"workers": 1, "max_in_flight": 1})
            result = collect_grid(serial, Path(td), FakeEpisodeBackend(delay_seconds=0.005))
            audit = audit_collection(Path(td))
            self.assertEqual(result["status"], "done")
            self.assertTrue(audit["pass"], audit)
            self.assertTrue(audit["checks"]["serial_execution_observed"])

    def test_two_configs_run_concurrently_without_mixing_or_updates(self):
        with tempfile.TemporaryDirectory() as td:
            backend = FakeEpisodeBackend(delay_seconds=0.04)
            result = collect_grid(BASE_CONFIG, Path(td), backend)
            audit = audit_collection(Path(td))

            self.assertEqual(result["status"], "done")
            self.assertTrue(audit["pass"], audit)
            self.assertEqual(audit["config_count"], 2)
            self.assertEqual(audit["episode_count"], 4)
            self.assertEqual(audit["duplicate_seed_count"], 0)
            self.assertEqual(audit["config_mixing_count"], 0)
            self.assertEqual(audit["optimizer_update_requests"], 0)
            self.assertGreaterEqual(audit["max_concurrency"], 2)
            self.assertGreater(audit["cross_config_overlap_seconds"], 0.0)

            records = [json.loads(line) for line in (Path(td) / "episodes.jsonl").read_text().splitlines()]
            self.assertEqual({r["worker_id"] for r in records}, {"fake-0", "fake-1"})
            self.assertTrue(all(r["schema"] == "mibot-grpo-rollout-v1" for r in records))
            self.assertTrue(all(r["trainer_payload"]["optimizer_update_requested"] is False for r in records))

    def test_resume_skips_completed_episode_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            first = collect_grid(BASE_CONFIG, out, FakeEpisodeBackend(), max_new_episodes=1)
            self.assertEqual(first["status"], "partial")
            first_record = (out / "episodes.jsonl").read_text().splitlines()[0]

            second = collect_grid(BASE_CONFIG, out, FakeEpisodeBackend())
            self.assertEqual(second["status"], "done")
            lines = (out / "episodes.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 4)
            self.assertEqual(lines[0], first_record)
            self.assertEqual(len({json.loads(line)["job_id"] for line in lines}), 4)

    def test_resume_rejects_corrupted_completed_row(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            collect_grid(BASE_CONFIG, out, FakeEpisodeBackend(), max_new_episodes=1)
            path = out / "episodes.jsonl"
            row = json.loads(path.read_text().splitlines()[0])
            row["seed"] += 1
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "resume row"):
                collect_grid(BASE_CONFIG, out, FakeEpisodeBackend())

    def test_worker_crash_retries_same_job_once_and_preserves_failure_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            backend = FakeEpisodeBackend(fail_once_seeds={100})
            result = collect_grid(BASE_CONFIG, Path(td), backend)
            self.assertEqual(result["status"], "done")
            self.assertEqual(result["retry_count"], 1)
            failures = [json.loads(line) for line in (Path(td) / "failures.jsonl").read_text().splitlines()]
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["seed"], 100)
            self.assertEqual(failures[0]["attempt"], 1)
            self.assertTrue((Path(td) / "DONE").exists())
            self.assertFalse((Path(td) / "FAILED").exists())

    def test_duplicate_launch_lock_refuses_second_owner(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            out.mkdir(exist_ok=True)
            (out / ".collector.lock").write_text(json.dumps({"pid": 999999, "run_id": "other"}))
            with self.assertRaisesRegex(RuntimeError, "collector lock"):
                collect_grid(BASE_CONFIG, out, FakeEpisodeBackend())

    def test_audit_mutation_catches_config_mixing(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            collect_grid(BASE_CONFIG, out, FakeEpisodeBackend())
            path = out / "episodes.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            other = next(row for row in rows if row["config_id"] != rows[0]["config_id"])
            rows[0]["config_hash"] = other["config_hash"]
            path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
            audit = audit_collection(out)
            self.assertFalse(audit["pass"])
            self.assertEqual(audit["config_mixing_count"], 1)

    def test_audit_mutation_catches_duplicate_seed(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            collect_grid(BASE_CONFIG, out, FakeEpisodeBackend(delay_seconds=0.01))
            path = out / "episodes.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[1]["seed"] = rows[0]["seed"]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            audit = audit_collection(out)
            self.assertFalse(audit["checks"]["no_duplicate_seeds"])

    def test_audit_mutation_catches_optimizer_request(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            collect_grid(BASE_CONFIG, out, FakeEpisodeBackend(delay_seconds=0.01))
            path = out / "episodes.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["trainer_payload"]["optimizer_update_requested"] = True
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            audit = audit_collection(out)
            self.assertFalse(audit["checks"]["no_optimizer_updates"])

    def test_audit_mutation_catches_missing_parallel_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            collect_grid(BASE_CONFIG, out, FakeEpisodeBackend(delay_seconds=0.01))
            path = out / "episodes.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for index, row in enumerate(rows):
                row["started_at"] = float(index * 2)
                row["finished_at"] = float(index * 2 + 1)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            audit = audit_collection(out)
            self.assertFalse(audit["checks"]["parallel_workers_observed"])
            self.assertFalse(audit["checks"]["cross_config_overlap_observed"])

    def test_production_audit_mutation_catches_payload_hash_mismatch(self):
        import hashlib
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            collect_grid(BASE_CONFIG, out, FakeEpisodeBackend(delay_seconds=0.01))
            manifest_path = out / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["runtime_backend"] = "rlinf-worker-group"
            manifest_path.write_text(json.dumps(manifest))
            rows_path = out / "episodes.jsonl"
            rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
            for index, row in enumerate(rows):
                payload = out / f"payload-{index}.pkl"
                video = out / f"video-{index}.mp4"
                payload.write_bytes(f"payload-{index}".encode())
                video.write_bytes(b"video")
                row["trainer_payload"].update({
                    "path": str(payload),
                    "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                })
                row["artifacts"] = {"trainer_store": str(payload), "video": str(video)}
            rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertTrue(audit_collection(out)["pass"])
            rows[0]["trainer_payload"]["sha256"] = "0" * 64
            rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            audit = audit_collection(out)
            self.assertFalse(audit["checks"]["trainer_payload_hashes_valid"])


if __name__ == "__main__":
    unittest.main()
