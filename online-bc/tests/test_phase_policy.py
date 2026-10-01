import importlib.util
import os
from pathlib import Path
import pickle
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from online_bc.models.phase_policy import PhasePolicy


class FakeBackend:
    def __init__(self):
        self.value = 0
        self.infer_fn = None

    def load(self, path):
        self.value = int(path)
        self.infer_fn = None

    def infer(self, sample, seed):
        if self.infer_fn is None:
            value = self.value
            self.infer_fn = lambda sample, seed: value + seed
        return self.infer_fn(sample, seed)


class PhaseRouting(unittest.TestCase):
    def test_interleaving_and_reload_keep_base_and_standard_separate(self):
        backend = FakeBackend()
        backend.infer({}, 0)
        policy = PhasePolicy(backend, backend.infer_fn)
        policy.load("5")
        self.assertEqual(policy.infer({}, 2, "base_prefix", "prefix"), (2, True))
        self.assertEqual(policy.infer({}, 2, "base_prefix", "cup_placement"), (7, False))
        self.assertEqual(policy.infer({}, 3, "base_prefix", "prefix"), (3, True))
        self.assertEqual(policy.infer({}, 2), (7, False))
        policy.load("6")
        self.assertEqual(policy.infer({}, 2), (8, False))
        self.assertEqual(policy.infer({}, 2, "base_prefix", "prefix"), (2, True))

    def test_requires_known_explicit_phase(self):
        policy = PhasePolicy(FakeBackend(), lambda sample, seed: seed)
        for variant, phase in [
            ("base_prefix", None),
            ("base_prefix", "button"),
            ("other", "prefix"),
        ]:
            with self.assertRaises(ValueError):
                policy.infer({}, 0, variant, phase)

    def test_client_rejects_missing_or_wrong_routing_receipt(self):
        helper = types.SimpleNamespace(
            rollout=types.SimpleNamespace(sample_history=lambda *args: [])
        )
        path = Path(__file__).resolve().parents[1] / "src/online_bc/rollout/policy_adapter.py"
        spec = importlib.util.spec_from_file_location("phase_client_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"online_bc.rollout.pan_common": helper}):
            spec.loader.exec_module(module)
        response = types.SimpleNamespace()

        class Context:
            def __enter__(self):
                return response

            def __exit__(self, *args):
                pass

        with patch.dict(
            os.environ,
            {"COFFEE_BC_POLICY_URL": "http://localhost:18317", "COFFEE_BC_VARIANT": "base_prefix"},
        ):
            client = module.Policy("pi05")
        replies = [
            dict(actions=np.zeros((50, 12)), version=5),
            dict(
                actions=np.zeros((50, 12)),
                version=5,
                variant="base_prefix",
                skill_phase="prefix",
                effective_policy_version=5,
            ),
            dict(
                actions=np.zeros((50, 12)),
                version=5,
                variant="base_prefix",
                skill_phase="prefix",
                effective_policy_version=0,
            ),
        ]
        requests = []

        def open_request(request, timeout):
            requests.append(pickle.loads(request.data))
            return Context()

        with patch.object(module.urllib.request, "urlopen", open_request):
            for reply in replies[:2]:
                response.read = lambda: pickle.dumps(reply)
                with self.assertRaises(AssertionError):
                    client.infer({}, {}, [], "full task", 1, 0, "prefix")
            response.read = lambda: pickle.dumps(replies[2])
            self.assertEqual(client.infer({}, {}, [], "full task", 1, 0, "prefix").shape, (16, 12))
        self.assertTrue(
            all(r["variant"] == "base_prefix" and r["skill_phase"] == "prefix" for r in requests)
        )


if __name__ == "__main__":
    unittest.main()
