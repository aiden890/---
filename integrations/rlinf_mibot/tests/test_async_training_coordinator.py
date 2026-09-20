from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from async_training_coordinator import (  # noqa: E402
    AmbiguousUpdateError,
    StateStore,
    make_batch,
    new_state,
    recovery_action,
    validate_batch,
    validate_lag,
    validate_rows_snapshot,
    validate_update_report,
)

SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import async_distributed_grasp_train as launcher  # noqa: E402


def snap(version, digest=None):
    return {"policy_version": version, "policy_hash": digest or f"hash-{version}"}


def ready_batch(version=7):
    batch = make_batch("batch-0001", {"version": version, "hash": f"hash-{version}"}, 1)
    batch["rollout_epochs"] = [
        {"index": 0, "run_id": "batch-0001-e0", "status": "ready"},
        {"index": 1, "run_id": "batch-0001-e1", "status": "ready"},
    ]
    batch["trainable_action_chunks_estimate"] = 1100
    batch["status"] = "ready"
    return batch


class AsyncCoordinatorStateTests(unittest.TestCase):
    def test_checkpoint_rotation_keeps_latest_and_every_fifth(self):
        state = {"history": [
            {"batch_id": f"batch-{version}",
             "snapshot": {"version": version, "hash": f"hash-{version}"}}
            for version in range(1, 8)
        ]}
        paths = launcher.learner_prune_paths(state, keep_every=5)
        self.assertIn("/results/batch-1/consume_smoke/checkpoint.pt", paths)
        self.assertNotIn("/results/batch-5/consume_smoke/checkpoint.pt", paths)
        self.assertNotIn("/results/batch-7/consume_smoke/checkpoint.pt", paths)
        self.assertEqual(sum("/adapter-v" in path for path in paths), 7)

    def test_production_preflight_requires_single_epoch_and_hash_history(self):
        metrics = {
            "policy_version": 7, "policy_hash": "hash-7",
            "policy_hash_history": {"7": "hash-7"}, "fail_stopped": False,
        }
        config = {"sampler": "pirl", "eta": 0.1, "lr": 5e-6,
                  "weight_decay": 0.01, "grad_clip": 1.0, "clip": 0.2,
                  "kl_coef": 0.0, "update_epochs": 1,
                  "lora_targets": "all_linear", "rank": 16, "alpha": 32}
        with mock.patch.object(launcher.legacy, "rpc", return_value={"config": config}):
            launcher.validate_production_servers(metrics, dict(metrics))
            advanced = dict(metrics, policy_version=8, policy_hash="hash-8",
                            policy_hash_history={"7": "hash-7", "8": "hash-8"})
            launcher.validate_production_servers(
                metrics, advanced, require_equal=False)
            with self.assertRaisesRegex(RuntimeError, "same policy snapshot"):
                launcher.validate_production_servers(metrics, advanced)
        config["update_epochs"] = 4
        with mock.patch.object(launcher.legacy, "rpc", return_value={"config": config}):
            with self.assertRaisesRegex(RuntimeError, "update_epochs"):
                launcher.validate_production_servers(metrics, dict(metrics))

    def test_batch_requires_contiguous_distinct_epochs_and_chunk_target(self):
        batch = ready_batch()
        validate_batch(batch)
        batch["rollout_epochs"][1]["index"] = 2
        with self.assertRaisesRegex(ValueError, "contiguous"):
            validate_batch(batch)
        batch = ready_batch()
        batch["trainable_action_chunks_estimate"] = 1023
        with self.assertRaisesRegex(ValueError, "target not met"):
            validate_batch(batch)

    def test_adaptive_chunk_estimate_drops_all_failure_groups(self):
        rows = []
        for group, successes in (("mixed", [False, True]),
                                 ("failed", [False, False])):
            for member, success in enumerate(successes):
                trajectory_id = f"run/{group}-{member}"
                rows.append({
                    "reward": float(success), "success": success, "steps": 32,
                    "trainer_payload": {
                        "group_id": group, "trajectory_ids": [trajectory_id],
                        "n_chunks": 7,
                    },
                })
        estimate = launcher.estimate_trainable_chunks(rows, 0)
        self.assertEqual(estimate["trainable_action_chunks"], 14)
        self.assertEqual(estimate["dropped_homogeneous_groups"],
                         ["rollout-0/failed"])

    def test_collection_stops_as_soon_as_chunk_target_is_met(self):
        batch = make_batch("adaptive", {"version": 7, "hash": "hash-7"}, 1)

        def rows_for_run(run_id, seed, *, groups, group_size, workers):
            del seed, groups, group_size
            self.assertEqual(workers, 4)
            return [{
                "reward": float(member), "success": bool(member), "steps": 32,
                "trainer_payload": {
                    "group_id": "group", "trajectory_ids": [f"{run_id}-t{member}"],
                    "n_chunks": 300, "policy_version": 7, "policy_hash": "hash-7",
                },
            } for member in range(2)]

        with mock.patch.object(launcher.legacy, "ssh",
                               return_value=SimpleNamespace(stdout="")), \
             mock.patch.object(launcher.legacy, "rpc", return_value=snap(7)), \
             mock.patch.object(launcher.legacy, "collect", side_effect=rows_for_run) as collect:
            result = launcher.collect_batch(batch)
        self.assertEqual(collect.call_count, 2)
        self.assertEqual(len(result["rollout_epochs"]), 2)
        self.assertEqual(result["trainable_action_chunks_estimate"], 1200)

    def test_rows_must_all_match_planned_snapshot(self):
        expected = {"version": 7, "hash": "hash-7"}
        rows = [{"trainer_payload": {"policy_version": 7, "policy_hash": "hash-7"}}
                for _ in range(8)]
        validate_rows_snapshot(rows, expected)
        rows[-1]["trainer_payload"]["policy_hash"] = "other"
        with self.assertRaisesRegex(ValueError, "snapshot mismatch"):
            validate_rows_snapshot(rows, expected)

    def test_only_lag_zero_or_one_is_accepted(self):
        batch = ready_batch(7)
        self.assertEqual(validate_lag(batch, {"version": 7, "hash": "hash-7"}), 0)
        self.assertEqual(validate_lag(batch, {"version": 8, "hash": "hash-8"}), 1)
        with self.assertRaisesRegex(ValueError, "lag"):
            validate_lag(batch, {"version": 9, "hash": "hash-9"})

    def test_report_binds_rollout_transition_and_reloadable_checkpoint(self):
        batch = ready_batch(7)
        report = {
            "status": "PASS", "checkpoint_reloadable": True,
            "checkpoint": "/results/batch/checkpoint.pt",
            "update": {
                "rollout_policy_version": 7, "policy_lag": 1,
                "policy_version_before": 8, "policy_version_after": 9,
                "policy_hash_after": "hash-9", "replayed_update": False,
            },
        }
        self.assertEqual(validate_update_report(
            report, batch, {"version": 8, "hash": "hash-8"}),
            {"version": 9, "hash": "hash-9"})
        report["checkpoint_reloadable"] = False
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            validate_update_report(report, batch, {"version": 8, "hash": "hash-8"})

    def test_recovery_never_retries_after_unreported_version_advance(self):
        state = new_state(snap(7), snap(7), 100)
        state["inflight"] = {"learner_before": {"version": 7, "hash": "hash-7"}}
        self.assertEqual(recovery_action(state, snap(7), False), "retry_same_update_id")
        self.assertEqual(recovery_action(state, snap(8), True), "commit_report")
        with self.assertRaises(AmbiguousUpdateError):
            recovery_action(state, snap(8), False)

    def test_atomic_state_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.json")
            state = new_state(snap(3), snap(3), 20)
            with store.locked():
                store.save(state)
                loaded = store.load()
            self.assertEqual(loaded, state)
            self.assertEqual(json.loads(store.path.read_text()), state)


if __name__ == "__main__":
    unittest.main()
