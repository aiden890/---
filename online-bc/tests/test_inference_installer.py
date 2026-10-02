"""Installer preflight and error handling without Docker or a running policy."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


path = Path(__file__).resolve().parents[1] / "scripts/install-inference-policy.py"
spec = importlib.util.spec_from_file_location("inference_installer", path)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallerGuards(unittest.TestCase):
    def test_timeout_does_not_expose_command_arguments(self):
        with patch.object(installer.subprocess, "run", side_effect=subprocess.TimeoutExpired(
                ["docker", "-e", "PRIVATE_VALUE=never-echo"], 1)):
            with self.assertRaises(RuntimeError) as caught:
                installer.command(["docker", "-e", "PRIVATE_VALUE=never-echo"])
        self.assertNotIn("never-echo", str(caught.exception))

    def test_failed_command_keeps_output_in_operation_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "operation.log"
            result = subprocess.CompletedProcess([], 1, "preserved stdout", "preserved stderr")
            with patch.object(installer.subprocess, "run", return_value=result):
                with self.assertRaises(RuntimeError):
                    installer.command(["docker"], log)
            self.assertEqual(log.read_text(), "preserved stdoutpreserved stderr")

    def test_wrong_checkpoint_is_rejected_before_actions(self):
        with patch.object(installer, "health", return_value=dict(
                model="pi05", ready=True, version=12, inference_only=True,
                base_prefix_enabled=True)):
            with self.assertRaises(AssertionError):
                installer.ready("unused", version=11, inference=True)

    def test_missing_inference_capability_is_rejected(self):
        with patch.object(installer, "health", return_value=dict(
                model="pi05", ready=True, version=11, base_prefix_enabled=True)):
            with self.assertRaises(AssertionError):
                installer.ready("unused", version=11, inference=True)

    def test_atomic_status_preserves_complete_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "status.json"
            installer.atomic(target, {"status": "waiting"})
            installer.atomic(target, {"status": "deferred", "production_changed": False})
            self.assertEqual(json.loads(target.read_text())["status"], "deferred")
            self.assertEqual(list(Path(tmp).iterdir()), [target])


if __name__ == "__main__":
    unittest.main()
