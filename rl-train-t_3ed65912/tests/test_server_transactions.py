"""Failure-injection tests for asynchronous policy identity and transactions.

These tests intentionally construct a tiny CPU server shell instead of loading the
MiBoT checkpoint.  They exercise the same mutation/rollback methods used by the GPU
server and therefore run in the production PyTorch image without CUDA memory cost.
"""
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import grpo_trainer_server as server_module  # noqa: E402
from grpo_trainer_server import GRPOTrainerServer  # noqa: E402
from update_batch import IdempotentUpdateCache  # noqa: E402


def tiny_server():
    server = GRPOTrainerServer.__new__(GRPOTrainerServer)
    server.model = torch.nn.Linear(2, 2, bias=False)
    server.model.device = torch.device("cpu")
    server.trainable_params = list(server.model.parameters())
    server.opt = torch.optim.AdamW(server.trainable_params, lr=0.1)
    server.wrappers = []
    server.store = {}
    server.update_cache = IdempotentUpdateCache()
    server.policy_version = 3
    server.policy_hash = server._compute_policy_hash()
    server.policy_hash_history = {2: "previous-hash", 3: server.policy_hash}
    server.fatal_error = None
    return server


def optimizer_state_copy(server):
    return GRPOTrainerServer._cpu_clone(server.opt.state_dict())


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


class ServerTransactionTests(unittest.TestCase):
    def test_update_exception_restores_weights_optimizer_and_policy_identity(self):
        server = tiny_server()
        before_weight = server.model.weight.detach().clone()
        before_optimizer = optimizer_state_copy(server)
        before_identity = (server.policy_version, server.policy_hash,
                           copy.deepcopy(server.policy_hash_history))
        server.store = {"trajectory": [{
            "policy_version": 3, "policy_hash": server.policy_hash,
        }]}

        def mutate_then_fail(_request):
            server.model.weight.grad = torch.ones_like(server.model.weight)
            server.opt.step()
            raise RuntimeError("injected post-step failure")

        with mock.patch.object(server, "_op_update_impl", side_effect=mutate_then_fail):
            with self.assertRaisesRegex(RuntimeError, "injected post-step failure"):
                server.op_update({"update_id": "failure-injection", "advantages": {}})
        self.assertTrue(torch.equal(server.model.weight, before_weight))
        assert_nested_equal(server.opt.state_dict(), before_optimizer)
        self.assertEqual(
            (server.policy_version, server.policy_hash, server.policy_hash_history),
            before_identity)
        self.assertEqual(server.store, {})
        self.assertIsNone(server.fatal_error)

    def test_update_rejects_version_with_unrecognized_hash(self):
        server = tiny_server()
        server.store = {"trajectory": [{
            "policy_version": 2, "policy_hash": "forged-hash",
        }]}
        with mock.patch.object(server, "_op_update_impl") as operation:
            with self.assertRaisesRegex(ValueError, "identity is not retained"):
                server.op_update({"update_id": "forged-identity", "advantages": {}})
        operation.assert_not_called()
        self.assertEqual(server.store, {})

    def test_adapter_hash_failure_is_transactional(self):
        server = tiny_server()
        before = server.model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-adapter.pt"
            torch.save({
                "schema": "grpo-adapter-v1", "lora": {},
                "extra_trainable": {"weight": torch.full_like(before, 9.0)},
                "policy_version": 4, "policy_hash": "intentionally-wrong",
            }, path)
            with self.assertRaisesRegex(RuntimeError, "adapter hash mismatch"):
                server.op_load_adapter({"path": str(path)})
        self.assertTrue(torch.equal(server.model.weight, before))
        self.assertEqual(server.policy_version, 3)
        self.assertEqual(server.policy_hash, server._compute_policy_hash())
        self.assertIsNone(server.fatal_error)

    def test_same_version_cannot_be_rebound_to_another_hash(self):
        server = tiny_server()
        before = server.model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rebound.pt"
            torch.save({
                "schema": "grpo-adapter-v1", "lora": {},
                "extra_trainable": {"weight": before + 1},
                "policy_version": 3, "policy_hash": "different-hash",
            }, path)
            with self.assertRaisesRegex(ValueError, "already names hash"):
                server.op_load_adapter({"path": str(path)})
        self.assertTrue(torch.equal(server.model.weight, before))

    def test_import_rejects_unknown_lag1_hash_and_removes_staged_rows(self):
        server = tiny_server()

        def fake_import(store, *args, **kwargs):
            del args, kwargs
            store["trajectory"] = [{"policy_version": 2, "policy_hash": "forged"}]
            return {"imported_trajectory_ids": ["trajectory"], "policy_version": 2,
                    "policy_hash": "forged"}

        with mock.patch.object(server_module, "import_rollout_store", fake_import):
            with self.assertRaisesRegex(ValueError, "unrecognized policy identity"):
                server.op_import_store({
                    "path": "/results/fake.pt", "expected_sha256": "unused"})
        self.assertEqual(server.store, {})

    def test_import_accepts_retained_lag1_identity(self):
        server = tiny_server()

        def fake_import(store, *args, **kwargs):
            del args, kwargs
            store["trajectory"] = [{"policy_version": 2, "policy_hash": "previous-hash"}]
            return {"imported_trajectory_ids": ["trajectory"], "policy_version": 2,
                    "policy_hash": "previous-hash", "policy_lag": 1}

        with mock.patch.object(server_module, "import_rollout_store", fake_import):
            result = server.op_import_store({
                "path": "/results/fake.pt", "expected_sha256": "unused",
                "max_policy_lag": 1})
        self.assertEqual(result["policy_lag"], 1)
        self.assertEqual(list(server.store), ["trajectory"])

    def test_terminal_chunk_masks_unexecuted_action_suffix(self):
        server = tiny_server()
        old = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        server.store = {"trajectory": [{
            "chunk_index": 7,
            "exec_mask_cpu": torch.ones(3, 4, dtype=torch.bool),
            "old_logp_elements_cpu": old,
            "old_logp": float(old.sum()),
        }]}
        result = server.op_set_last_chunk_executed_steps({
            "trajectory_id": "trajectory", "executed_steps": 2})
        chunk = server.store["trajectory"][-1]
        self.assertEqual(result["masked_suffix_steps"], 1)
        self.assertTrue(torch.equal(
            chunk["exec_mask_cpu"],
            torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.bool)))
        self.assertEqual(chunk["old_logp"], float(old[:2].sum()))

    def test_terminal_chunk_mask_cannot_expand_or_remove_whole_chunk(self):
        server = tiny_server()
        server.store = {"trajectory": [{
            "exec_mask_cpu": torch.ones(3, 2, dtype=torch.bool),
            "old_logp_elements_cpu": torch.zeros(3, 2),
            "old_logp": 0.0,
        }]}
        for invalid in (0, 4):
            with self.subTest(executed_steps=invalid), self.assertRaises(ValueError):
                server.op_set_last_chunk_executed_steps({
                    "trajectory_id": "trajectory", "executed_steps": invalid})


if __name__ == "__main__":
    unittest.main()
