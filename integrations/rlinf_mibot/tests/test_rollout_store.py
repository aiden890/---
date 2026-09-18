from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    from rollout_store import export_rollout_store, import_rollout_store  # noqa: E402


@unittest.skipUnless(TORCH_AVAILABLE, "torch is provided by the production GPU image")
class RolloutStoreTests(unittest.TestCase):
    def test_export_then_import_restores_selected_trajectories(self):
        source = {"traj-a": [{"old_logp": 1.0}], "traj-b": [{"old_logp": 2.0}]}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "payload.pt"
            meta = export_rollout_store(source, ["traj-b"], path)
            target = {}
            loaded = import_rollout_store(target, path, expected_sha256=meta["sha256"])
        self.assertEqual(meta["trajectory_ids"], ["traj-b"])
        self.assertEqual(meta["optimizer_update_requested"], False)
        self.assertEqual(loaded["imported_trajectory_ids"], ["traj-b"])
        self.assertEqual(target, {"traj-b": [{"old_logp": 2.0}]})

    def test_export_rejects_missing_trajectory(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(KeyError, "missing"):
                export_rollout_store({}, ["missing"], Path(td) / "payload.pt")

    def test_import_rejects_duplicate_trajectory(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "payload.pt"
            meta = export_rollout_store({"traj-a": [{"x": 1}]}, ["traj-a"], path)
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                import_rollout_store({"traj-a": [{"x": 2}]}, path,
                                     expected_sha256=meta["sha256"])

    def test_import_rejects_hash_mismatch_without_mutating_store(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "payload.pt"
            export_rollout_store({"traj-a": [{"x": 1}]}, ["traj-a"], path)
            target = {}
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                import_rollout_store(target, path, expected_sha256="0" * 64)
            self.assertEqual(target, {})


if __name__ == "__main__":
    unittest.main()
