from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consume_collector import build_smoke_advantages, load_collector_payloads, validate_update  # noqa: E402


class ConsumeCollectorTests(unittest.TestCase):
    def test_update_gate_catches_nonfinite_mutation(self):
        clean = {
            "adapter_delta_l2": 0.25,
            "n_nonfinite": 0,
            "n_dropped": 0,
            "epoch_stats": [{"epoch": 0, "mean_ratio": 1.0}],
            "post_step_n_nonfinite": 0,
            "post_step_n_dropped": 0,
        }
        self.assertTrue(validate_update(clean)["pass"])
        mutated = dict(clean, post_step_n_nonfinite=1)
        result = validate_update(mutated)
        self.assertFalse(result["pass"])
        self.assertFalse(result["checks"]["no_nonfinite"])

    def test_update_gate_catches_replayed_update_mutation(self):
        update = {
            "adapter_delta_l2": 0.25, "n_nonfinite": 0, "n_dropped": 0,
            "epoch_stats": [{"epoch": 0, "mean_ratio": 1.0}],
            "post_step_n_nonfinite": 0, "post_step_n_dropped": 0,
            "replayed_update": True,
        }
        result = validate_update(update)
        self.assertFalse(result["pass"])
        self.assertFalse(result["checks"]["new_update_not_replay"])

    def test_loads_payloads_and_builds_nonzero_advantages(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            payload_a = out / "a.pkl"; payload_a.write_bytes(b"a")
            payload_b = out / "b.pkl"; payload_b.write_bytes(b"b")
            rows = [
                {"job_id": "a", "reward": 0.0,
                 "trainer_payload": {"path": str(payload_a),
                                     "sha256": hashlib.sha256(b"a").hexdigest(),
                                     "trajectory_ids": ["a"],
                                     "optimizer_update_requested": False}},
                {"job_id": "b", "reward": 1.0,
                 "trainer_payload": {"path": str(payload_b),
                                     "sha256": hashlib.sha256(b"b").hexdigest(),
                                     "trajectory_ids": ["b"],
                                     "optimizer_update_requested": False}},
            ]
            (out / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            payloads = load_collector_payloads(out)
            advantages = build_smoke_advantages(payloads)
        self.assertEqual([item["path"] for item in payloads], [str(payload_a), str(payload_b)])
        self.assertEqual(advantages, {"a": -1.0, "b": 1.0})

    def test_equal_rewards_still_produce_bounded_mutation_smoke_signal(self):
        payloads = [
            {"path": "a", "reward": 0.0, "trajectory_ids": ["a"]},
            {"path": "b", "reward": 0.0, "trajectory_ids": ["b"]},
        ]
        self.assertEqual(build_smoke_advantages(payloads), {"a": -1.0, "b": 1.0})

    def test_rejects_collector_payload_that_requested_update(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            row = {"reward": 1.0, "trainer_payload": {"path": "x", "trajectory_ids": ["x"],
                                                        "optimizer_update_requested": True}}
            (out / "episodes.jsonl").write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "optimizer"):
                load_collector_payloads(out)

    def test_rejects_payload_path_outside_run_directory(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "run"; out.mkdir()
            outside = Path(td) / "outside.pkl"; outside.write_bytes(b"unsafe")
            row = {"reward": 1.0, "trainer_payload": {
                "path": str(outside), "sha256": hashlib.sha256(b"unsafe").hexdigest(),
                "trajectory_ids": ["x"], "optimizer_update_requested": False}}
            (out / "episodes.jsonl").write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "outside run directory"):
                load_collector_payloads(out)


if __name__ == "__main__":
    unittest.main()
