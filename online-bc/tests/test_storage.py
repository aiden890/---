import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "rollout_archive", Path(__file__).resolve().parents[1] / "scripts/archive-rollout-data.py"
)
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


class ArchiveEligibility(unittest.TestCase):
    def test_keeps_recent_rounds_and_requires_completed_uploaded_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "current-adapter.json").write_text(json.dumps({"version": 5}))
            for number in range(1, 6):
                batch = root / "rollouts" / f"round-{number:04d}" / "batch-01"
                (batch / "pi05").mkdir(parents=True)
                (batch / "upload").mkdir()
                (batch / "pi05/complete.json").write_text("{}")
                if number != 2:
                    (batch / "upload/manifest.json").write_text("{}")
            incomplete = root / "rollouts/round-0003/batch-02"
            incomplete.mkdir()
            self.assertEqual([p.name for p in archive.eligible(root)], ["round-0001"])


if __name__ == "__main__":
    unittest.main()
