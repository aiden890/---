from __future__ import annotations

import json
import hashlib
import importlib.util
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consume_collector import (TrainerRPC, build_smoke_advantages,
                               filter_homogeneous_groups,
                               inspect_action_chunks, load_collector_payloads,
                               optimizer_update_epochs,
                               require_trainable_action_chunks,
                               validate_rollout_batch, validate_update)  # noqa: E402

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class ConsumeCollectorTests(unittest.TestCase):
    def test_z1_rollout_epochs_are_distinct_from_optimizer_epochs(self):
        payloads = []
        for epoch in range(4):
            for member in range(8):
                payloads.append({"group_id": f"epoch-{epoch}-group-0",
                                 "rollout_epoch": epoch,
                                 "trajectory_ids": [f"e{epoch}-m{member}"]})
        diagnostics = validate_rollout_batch(
            payloads, expected_group_size=8, min_rollout_epochs=4)
        self.assertEqual(diagnostics["rollout_epochs_collected"], 4)
        self.assertEqual(optimizer_update_epochs({"optimizer_update_epochs": 1}), 1)

    def test_reused_group_names_across_rollout_epochs_stay_independent(self):
        payloads = []
        for epoch in range(2):
            payloads.extend([
                {"group_id": "group-0", "rollout_epoch": epoch, "reward": 0.0,
                 "trajectory_ids": [f"loss-{epoch}"]},
                {"group_id": "group-0", "rollout_epoch": epoch, "reward": 1.0,
                 "trajectory_ids": [f"win-{epoch}"]},
            ])
        diagnostics = validate_rollout_batch(payloads, expected_group_size=2)
        self.assertEqual(diagnostics["group_count_collected"], 2)
        self.assertEqual(build_smoke_advantages(payloads), {
            "loss-0": -1.0, "win-0": 1.0, "loss-1": -1.0, "win-1": 1.0})

    def test_rejects_too_few_fresh_rollout_epochs_and_wrong_group_size(self):
        payloads = [{"group_id": "g", "rollout_epoch": 0,
                     "trajectory_ids": [str(index)]} for index in range(7)]
        with self.assertRaisesRegex(ValueError, "exactly 8"):
            validate_rollout_batch(payloads, expected_group_size=8)
        with self.assertRaisesRegex(ValueError, "at least 4 fresh rollout epochs"):
            validate_rollout_batch(payloads, min_rollout_epochs=4)

    def test_missing_rollout_epoch_provenance_is_rejected_for_production(self):
        payloads = [{"group_id": "g", "rollout_epoch": None,
                     "trajectory_ids": [str(index)]} for index in range(8)]
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_rollout_batch(payloads, min_rollout_epochs=4)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is provided by the production GPU image")
    def test_actual_payload_count_excludes_zero_advantage_chunks(self):
        import torch
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "payload.pt"
            torch.save({
                "schema": "grpo-trainer-store-v2",
                "trajectories": {
                    "train": [{"x": 1}, {"x": 2}],
                    "baseline": [{"x": 3}, {"x": 4}, {"x": 5}],
                },
            }, path)
            payloads = [{
                "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "group_id": "g", "trajectory_ids": ["train", "baseline"],
            }]
            diagnostics = inspect_action_chunks(
                payloads, {"train": 0.75, "baseline": 0.0})
        self.assertEqual(diagnostics["action_chunks_total"], 5)
        self.assertEqual(diagnostics["trainable_action_chunks"], 2)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            require_trainable_action_chunks(diagnostics, 3)
        require_trainable_action_chunks(diagnostics, 2)

    def test_payload_counting_logic_without_gpu_runtime(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "payload.pt"
            path.write_bytes(pickle.dumps({
                "schema": "grpo-trainer-store-v2",
                "trajectories": {"active": [{}, {}], "zero": [{}, {}, {}]},
            }))
            fake_torch = SimpleNamespace(
                load=lambda stream, map_location, weights_only: pickle.load(stream))
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                diagnostics = inspect_action_chunks([{
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "group_id": "g", "trajectory_ids": ["active", "zero"],
                }], {"active": 1.0, "zero": 0.0})
        self.assertEqual(diagnostics["action_chunks_total"], 5)
        self.assertEqual(diagnostics["trainable_action_chunks"], 2)

    def test_optimizer_epoch_alias_conflict_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            optimizer_update_epochs({"optimizer_update_epochs": 1, "update_epochs": 4})

    def test_minimum_trainable_chunk_gate(self):
        diagnostics = {"trainable_action_chunks": 1023}
        with self.assertRaisesRegex(ValueError, "at least 1024"):
            require_trainable_action_chunks(diagnostics, 1024)
        diagnostics["trainable_action_chunks"] = 1024
        require_trainable_action_chunks(diagnostics, 1024)

    def test_rpc_connect_timeout_does_not_limit_gpu_update_lifetime(self):
        sock = mock.Mock()
        with mock.patch("consume_collector.socket.create_connection",
                        return_value=sock) as connect:
            client = TrainerRPC("learner", 10088)
        connect.assert_called_once_with(("learner", 10088), timeout=60)
        sock.settimeout.assert_called_once_with(None)
        client.close()

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

    def test_lag_one_update_does_not_require_ratio_exactly_one(self):
        update = {
            "adapter_delta_l2": 0.25, "n_nonfinite": 0, "n_dropped": 0,
            "epoch_stats": [{"epoch": 0, "mean_ratio": 0.93}],
            "post_step_n_nonfinite": 0, "post_step_n_dropped": 0,
            "policy_lag": 1, "replayed_update": False,
        }
        result = validate_update(update)
        self.assertTrue(result["pass"])
        self.assertTrue(result["checks"]["epoch0_ratio_valid"])

    def test_lag_zero_allows_bounded_cross_gpu_bf16_drift(self):
        base = {
            "adapter_delta_l2": 0.25, "n_nonfinite": 0, "n_dropped": 0,
            "post_step_n_nonfinite": 0, "post_step_n_dropped": 0,
            "policy_lag": 0, "replayed_update": False,
        }
        self.assertTrue(validate_update(
            dict(base, epoch_stats=[{"epoch": 0, "mean_ratio": 0.99994}]))["pass"])
        self.assertFalse(validate_update(
            dict(base, epoch_stats=[{"epoch": 0, "mean_ratio": 0.99}]))["pass"])

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
        self.assertEqual([item["path"] for item in payloads],
                         [str(payload_a.resolve()), str(payload_b.resolve())])
        self.assertEqual(advantages, {"a": -1.0, "b": 1.0})

    def test_equal_rewards_still_produce_bounded_mutation_smoke_signal(self):
        payloads = [
            {"path": "a", "reward": 0.0, "trajectory_ids": ["a"]},
            {"path": "b", "reward": 0.0, "trajectory_ids": ["b"]},
        ]
        self.assertEqual(build_smoke_advantages(payloads), {"a": -1.0, "b": 1.0})

    def test_advantages_are_normalized_within_each_group(self):
        payloads = [
            {"group_id": "g1", "reward": 0.0, "trajectory_ids": ["a"]},
            {"group_id": "g1", "reward": 1.0, "trajectory_ids": ["b"]},
            {"group_id": "g2", "reward": 10.0, "trajectory_ids": ["c"]},
            {"group_id": "g2", "reward": 20.0, "trajectory_ids": ["d"]},
        ]
        self.assertEqual(build_smoke_advantages(payloads),
                         {"a": -1.0, "b": 1.0, "c": -1.0, "d": 1.0})

    def test_homogeneous_groups_are_filtered(self):
        payloads = [
            {"group_id": "drop", "reward": 0.0}, {"group_id": "drop", "reward": 0.0},
            {"group_id": "keep", "reward": 0.0}, {"group_id": "keep", "reward": 1.0},
        ]
        kept, dropped = filter_homogeneous_groups(payloads)
        self.assertEqual([item["group_id"] for item in kept], ["keep", "keep"])
        self.assertEqual(dropped, ["drop"])

    def test_success_signal_recovers_zero_terminal_rewards(self):
        payloads = [
            {"group_id": "mixed", "reward": 0.0, "success": False,
             "trajectory_ids": ["a"]},
            {"group_id": "mixed", "reward": 0.0, "success": True,
             "trajectory_ids": ["b"]},
        ]
        kept, dropped = filter_homogeneous_groups(payloads)
        self.assertEqual(kept, payloads)
        self.assertEqual(dropped, [])
        self.assertEqual(build_smoke_advantages(payloads), {"a": -1.0, "b": 1.0})

    def test_all_success_group_is_filtered_even_when_shaped_rewards_differ(self):
        payloads = [
            {"group_id": "all-win", "reward": 0.25, "success": True},
            {"group_id": "all-win", "reward": 1.00, "success": True},
        ]
        kept, dropped = filter_homogeneous_groups(payloads)
        self.assertEqual(kept, [])
        self.assertEqual(dropped, ["all-win"])

    def test_z1_all_success_uses_nonnegative_completion_baseline(self):
        payloads = [
            {"group_id": "all-win", "reward": 1.0, "success": True,
             "steps": 16, "trajectory_ids": ["fast"]},
            {"group_id": "all-win", "reward": 1.0, "success": True,
             "steps": 32, "trajectory_ids": ["slow"]},
        ]
        kept, dropped = filter_homogeneous_groups(payloads, retain_all_success=True)
        self.assertEqual(kept, payloads)
        self.assertEqual(dropped, [])
        advantages = build_smoke_advantages(payloads, success_decay_gamma=0.998)
        self.assertGreater(advantages["fast"], 0.0)
        self.assertEqual(advantages["slow"], 0.0)

    def test_z1_mixed_group_calibrates_success_by_action_chunks(self):
        payloads = [
            {"group_id": "mixed", "reward": 1.0, "success": True,
             "steps": 16, "trajectory_ids": ["win"]},
            {"group_id": "mixed", "reward": 0.0, "success": False,
             "steps": 208, "trajectory_ids": ["loss"]},
        ]
        advantages = build_smoke_advantages(payloads, success_decay_gamma=0.998)
        self.assertAlmostEqual(advantages["win"], 1.0)
        self.assertAlmostEqual(advantages["loss"], -1.0)

    def test_incomplete_success_metadata_falls_back_to_reward_variance(self):
        payloads = [
            {"group_id": "legacy", "reward": 0.0, "success": False},
            {"group_id": "legacy", "reward": 1.0},
        ]
        kept, dropped = filter_homogeneous_groups(payloads)
        self.assertEqual(kept, payloads)
        self.assertEqual(dropped, [])

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

    def test_rejects_nonfinite_reward_and_non_boolean_success(self):
        for reward, success, message in ((float("nan"), False, "not finite"),
                                         (0.0, "false", "boolean or null")):
            with self.subTest(reward=reward, success=success), tempfile.TemporaryDirectory() as td:
                out = Path(td)
                rows = []
                for index in range(2):
                    path = out / f"{index}.pkl"; path.write_bytes(str(index).encode())
                    rows.append({"reward": reward, "success": success,
                                 "trainer_payload": {
                                     "path": str(path),
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                     "trajectory_ids": [str(index)],
                                     "optimizer_update_requested": False}})
                (out / "episodes.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaisesRegex(ValueError, message):
                    load_collector_payloads(out)

    def test_rejects_duplicate_trajectory_ids(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            rows = []
            for index in range(2):
                path = out / f"{index}.pkl"; path.write_bytes(str(index).encode())
                rows.append({"reward": float(index), "success": bool(index),
                             "trainer_payload": {
                                 "path": str(path),
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "trajectory_ids": ["duplicate"],
                                 "optimizer_update_requested": False}})
            (out / "episodes.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "duplicate collector trajectory IDs"):
                load_collector_payloads(out)


if __name__ == "__main__":
    unittest.main()
