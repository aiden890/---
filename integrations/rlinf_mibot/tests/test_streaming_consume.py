from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streaming_consume as subject  # noqa: E402


class FakeRPC:
    calls = []

    def __init__(self, host, port):
        pass

    def call(self, request):
        self.calls.append(request)
        if request["op"] == "reset":
            return {"reset": True}
        if request["op"] == "import_store":
            return {"imported": request["path"]}
        raise AssertionError(request)

    def close(self):
        pass


class StreamingConsumeTests(unittest.TestCase):
    def test_epoch_is_imported_before_learner_files_are_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = root / "batch"
            run = root / "run"
            payload_files = []
            payloads = []
            for member in range(8):
                path = run / f"p{member}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"payload")
                payload_files.append(path)
                payloads.append({
                    "path": str(path), "sha256": "digest", "group_id": "group",
                    "trajectory_ids": [f"traj-{member}"], "policy_version": 4,
                    "policy_hash": "hash-4", "success": member == 0,
                    "reward": float(member == 0), "steps": 16,
                })
            FakeRPC.calls = []
            with mock.patch.object(subject, "TrainerRPC", FakeRPC):
                subject.begin(batch, "host", 1, {"version": 4, "hash": "hash-4"})
                with mock.patch.object(subject, "load_collector_payloads",
                                       return_value=payloads), \
                     mock.patch.object(subject, "inspect_action_chunks", return_value={
                         "trainable_action_chunks": 8, "action_chunks_total": 8}), \
                     mock.patch.object(subject, "build_smoke_advantages", return_value={
                         f"traj-{member}": (-1.0 if member else 1.0)
                         for member in range(8)}):
                    result = subject.stage_epoch(
                        batch, run, 0, "host", 1,
                        {"expected_group_size": 8, "success_decay_gamma": 0.998})
            self.assertEqual(len([c for c in FakeRPC.calls if c["op"] == "import_store"]), 8)
            self.assertTrue(all(not path.exists() for path in payload_files))
            self.assertEqual(result["rollout_epochs"][0]["index"], 0)

    def test_epoch_order_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory) / "batch"
            FakeRPC.calls = []
            with mock.patch.object(subject, "TrainerRPC", FakeRPC):
                subject.begin(batch, "host", 1, {"version": 1, "hash": "hash-1"})
            with self.assertRaisesRegex(ValueError, "in order"):
                subject.stage_epoch(batch, Path(directory), 1, "host", 1,
                                    {"expected_group_size": 8})

    def test_four_epochs_accumulate_then_finalize_with_one_optimizer_epoch(self):
        class FullRPC(FakeRPC):
            def call(self, request):
                self.calls.append(request)
                op = request["op"]
                if op == "reset":
                    return {"reset": True}
                if op == "import_store":
                    return {"imported": True}
                if op == "metrics":
                    return {"policy_version": 5, "policy_hash": "hash-5"}
                if op == "update":
                    return {
                        "adapter_delta_l2": 0.1, "n_nonfinite": 0, "n_dropped": 0,
                        "post_step_n_nonfinite": 0, "post_step_n_dropped": 0,
                        "epoch_stats": [{"epoch": 0, "mean_ratio": 1.0}],
                        "policy_lag": 0, "rollout_policy_version": 5,
                        "policy_version_before": 5, "policy_version_after": 6,
                        "policy_hash_after": "hash-6", "replayed_update": False,
                    }
                if op == "save":
                    Path(request["path"]).write_bytes(b"checkpoint")
                    return {"saved": True}
                if op == "load":
                    return {"loaded": True}
                raise AssertionError(request)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = root / "batch"
            epoch_payloads = []
            for epoch in range(4):
                values = []
                for member in range(8):
                    path = root / f"e{epoch}-m{member}.pt"
                    path.write_bytes(b"payload")
                    values.append({
                        "path": str(path), "sha256": "digest", "group_id": "group",
                        "trajectory_ids": [f"e{epoch}-t{member}"], "policy_version": 5,
                        "policy_hash": "hash-5", "success": member == 0,
                        "reward": float(member == 0), "steps": 16,
                    })
                epoch_payloads.append(values)
            FullRPC.calls = []
            with mock.patch.object(subject, "TrainerRPC", FullRPC), \
                 mock.patch.object(subject, "load_collector_payloads",
                                   side_effect=epoch_payloads), \
                 mock.patch.object(subject, "inspect_action_chunks", return_value={
                     "trainable_action_chunks": 256, "action_chunks_total": 256}):
                subject.begin(batch, "host", 1, {"version": 5, "hash": "hash-5"})
                for epoch in range(4):
                    subject.stage_epoch(batch, root, epoch, "host", 1, {
                        "expected_group_size": 8, "success_decay_gamma": 0.998})
                report = subject.finalize(batch, "host", 1, {
                    "expected_group_size": 8, "min_rollout_epochs": 4,
                    "min_trainable_action_chunks": 1024,
                    "optimizer_update_epochs": 1, "success_decay_gamma": 0.998,
                    "clip": 0.2, "kl_coef": 0.0, "target_kl": None,
                })
            resets = [item for item in FullRPC.calls if item["op"] == "reset"]
            updates = [item for item in FullRPC.calls if item["op"] == "update"]
            self.assertEqual(len(resets), 1)
            self.assertEqual(len(updates), 1)
            self.assertEqual(updates[0]["update_epochs"], 1)
            self.assertEqual(len(updates[0]["advantages"]), 32)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(set(report["stage_timings"]), {
                "metrics_seconds", "optimizer_update_seconds",
                "checkpoint_save_seconds", "checkpoint_reload_seconds",
            })
            self.assertTrue(all(value >= 0 for value in report["stage_timings"].values()))
            self.assertEqual(
                report["algorithm_diagnostics"]["collection"]["rollout_epoch_ids"],
                [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
