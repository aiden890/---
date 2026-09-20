from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_streaming_transfer import prepare  # noqa: E402


class PrepareStreamingTransferTests(unittest.TestCase):
    def test_hardlinks_only_groups_that_can_produce_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            rows = []
            for group, outcomes in (("failed", [False, False]),
                                    ("mixed", [False, True])):
                for member, success in enumerate(outcomes):
                    payload = run / group / "payloads" / f"{member}.pt"
                    payload.parent.mkdir(parents=True, exist_ok=True)
                    payload.write_bytes(f"{group}-{member}".encode())
                    rows.append({
                        "success": success, "reward": float(success),
                        "trainer_payload": {"group_id": group, "path": str(payload)},
                    })
            (run / "episodes.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows))
            stage = root / ".transfer-run"
            result = prepare(run, stage, allowed_root=root)
            transferred = [json.loads(line) for line in
                           (stage / "episodes.jsonl").read_text().splitlines()]
            self.assertEqual(result["groups_transferred"], 1)
            self.assertEqual(result["episodes_transferred"], 2)
            self.assertTrue(all(row["trainer_payload"]["group_id"] == "mixed"
                                for row in transferred))
            source = run / "mixed" / "payloads" / "0.pt"
            linked = stage / "mixed" / "payloads" / "0.pt"
            self.assertEqual(os.stat(source).st_ino, os.stat(linked).st_ino)
            self.assertFalse((stage / "failed").exists())


if __name__ == "__main__":
    unittest.main()
