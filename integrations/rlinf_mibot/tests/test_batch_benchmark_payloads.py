from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_batch_benchmark_payloads import main, parse_sizes  # noqa: E402


class BatchBenchmarkPayloadTests(unittest.TestCase):
    def _write_source(self, root: Path, *, successes=(False, True), outside=False) -> Path:
        source = root / "source"
        source.mkdir()
        payload_dir = root / "outside" if outside else source / "payloads"
        payload_dir.mkdir()
        rows = []
        for index, success in enumerate(successes):
            trajectory_id = f"trajectory-{index}"
            payload_path = payload_dir / f"{trajectory_id}.pt"
            torch.save({"trajectories": {trajectory_id: [{"chunk": index}]},
                        "group_id": "group-0"}, payload_path)
            rows.append({
                "config_id": "cfg-0", "group_id": "group-0",
                "job_id": f"job-{index}", "reward": float(index),
                "success": success,
                "trainer_payload": {
                    "path": str(payload_path),
                    "sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
                    "trajectory_ids": [trajectory_id], "n_chunks": 1,
                    "group_id": "group-0",
                },
                "artifacts": {"trainer_store": str(payload_path)},
            })
        (source / "episodes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows))
        return source

    def test_parse_sizes_rejects_zero_negative_and_duplicates(self):
        for raw in ("", "0", "-1", "1,1", "1,,2", "1,"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_sizes(raw)
        self.assertEqual(parse_sizes("1, 2,4"), [1, 2, 4])

    def test_generates_isolated_groups_and_valid_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._write_source(root)
            output = root / "out"
            with patch.object(sys, "argv", ["build", str(source), str(output),
                                             "--sizes", "1,2"]):
                main()
            for size in (1, 2):
                destination = output / f"batchbench-g{size}"
                rows = [json.loads(line) for line in
                        (destination / "episodes.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), 2 * size)
                self.assertEqual(len({row["group_id"] for row in rows}), size)
                for row in rows:
                    payload_path = Path(row["trainer_payload"]["path"])
                    self.assertTrue(payload_path.is_relative_to(destination))
                    self.assertEqual(hashlib.sha256(payload_path.read_bytes()).hexdigest(),
                                     row["trainer_payload"]["sha256"])

    def test_rejects_all_success_despite_shaped_reward_variance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._write_source(root, successes=(True, True))
            with patch.object(sys, "argv", ["build", str(source), str(root / "out")]):
                with self.assertRaisesRegex(ValueError, "homogeneous"):
                    main()

    def test_rejects_source_payload_outside_source_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._write_source(root, outside=True)
            with patch.object(sys, "argv", ["build", str(source), str(root / "out")]):
                with self.assertRaisesRegex(ValueError, "outside source directory"):
                    main()

    def test_rejects_destination_escape(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._write_source(root)
            with patch.object(sys, "argv", ["build", str(source), str(root / "out"),
                                             "--prefix", "../escape"]):
                with self.assertRaisesRegex(ValueError, "escapes output root"):
                    main()


if __name__ == "__main__":
    unittest.main()
