import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from online_bc.models.phase_policy import PhasePolicy
from online_bc.models.serve_bc_policy import PolicyService
from online_bc.rollout.worker import inference_checkpoint_download
from online_bc.rollout.docker_worker import policy_options


class Backend:
    def __init__(self):
        self.calls = []
        self.infer_fn = None

    def load(self, path, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("load_optimizer", True) and not (Path(path) / "pi05-optimizer.pkl").exists():
            raise FileNotFoundError("optimizer")
        self.infer_fn = None

    def infer(self, sample, seed):
        return np.zeros((50, 12))


class InferenceServerTest(unittest.TestCase):
    def test_docker_restart_preserves_explicit_capabilities_only(self):
        self.assertEqual(policy_options({}, "pi05"), [])
        options = dict(OnlineBCPolicy=dict(inference_only=True, enable_base_prefix=True))
        self.assertEqual(policy_options(options, "pi05"), ["--inference-only", "--enable-base-prefix"])
        with self.assertRaises(ValueError):
            policy_options(options, "groot")
        with self.assertRaises(AssertionError):
            policy_options(dict(OnlineBCPolicy=dict(inference_only="false")), "pi05")

    def test_initial_reload_and_phase_cache_with_missing_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "metadata.json").write_text(json.dumps(dict(model="pi05", step=10)))
            backend = Backend()
            default = PolicyService("pi05", backend)
            with self.assertRaises(FileNotFoundError):
                default.load(path)
            self.assertEqual(default.version, 0)
            service = PolicyService("pi05", backend, inference_only=True)
            service.phase_policy = PhasePolicy(backend, object())
            service.phase_policy.adapter_snapshot = object()
            self.assertTrue(service.load(path)["inference_only"])
            self.assertIsNone(service.phase_policy.adapter_snapshot)
            self.assertEqual(backend.calls[-1], dict(load_optimizer=False))
            (path / "metadata.json").write_text(json.dumps(dict(model="pi05", step=11)))
            self.assertEqual(service.request(dict(op="load", path=str(path)))["version"], 11)
            self.assertTrue(service.health()["inference_only"])
            self.assertEqual(service.request(dict(sample={}, seed=5))["actions"].shape, (50, 12))
            (path / "metadata.json").write_text(json.dumps(dict(model="groot", step=12)))
            with self.assertRaises(AssertionError):
                service.load(path)
            self.assertEqual(service.version, 11)
        with self.assertRaises(ValueError):
            PolicyService("groot", backend, inference_only=True)

    def test_worker_requires_running_capability_before_filtering(self):
        config = dict(model="pi05", inference_only_download=True)
        for health in [{}, dict(model="pi05", ready=True, inference_only=False),
                       dict(model="groot", ready=True, inference_only=True)]:
            with self.assertRaises(RuntimeError):
                inference_checkpoint_download(config, health)
        self.assertTrue(inference_checkpoint_download(
            config, dict(model="pi05", ready=True, inference_only=True)))
        self.assertFalse(inference_checkpoint_download(dict(model="pi05"), {}))

    def test_transport_filter_is_download_only_and_default_preserves_optimizer(self):
        calls = []
        required = ["metadata.json", "pi05-lora.npz", "updates.json"]

        def sync(source, destination, **kwargs):
            calls.append(kwargs)
            for name in kwargs.get("include", required + ["pi05-optimizer.pkl"]):
                (Path(destination) / name).write_text("test")

        hub = types.ModuleType("huggingface_hub")
        hub.sync_bucket = sync
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"huggingface_hub": hub}):
            root = Path(tmp)
            token = root / "token"
            token.write_text("test-only")
            argv = ["hf_transfer", "download", str(root / "filtered"),
                    "test/weights/pi05/round-0010", "--token-file", str(token)]
            with patch.object(sys, "argv", argv + ["--inference-only"]), contextlib.redirect_stdout(io.StringIO()):
                runpy.run_module("online_bc.transport.hf_transfer", run_name="__main__")
            self.assertEqual(calls[-1]["include"], required)
            self.assertFalse((root / "filtered/pi05-optimizer.pkl").exists())
            argv[2] = str(root / "full")
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                runpy.run_module("online_bc.transport.hf_transfer", run_name="__main__")
            self.assertNotIn("include", calls[-1])
            self.assertTrue((root / "full/pi05-optimizer.pkl").exists())
            for direction, prefix in [("upload", "test/weights/pi05/round-0010"),
                                      ("download", "test/data/pi05/round-0010")]:
                argv[1], argv[3] = direction, prefix
                with patch.object(sys, "argv", argv + ["--inference-only"]), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        runpy.run_module("online_bc.transport.hf_transfer", run_name="__main__")
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
