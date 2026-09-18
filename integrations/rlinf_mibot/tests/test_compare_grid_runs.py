from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compare_grid_runs import summarize_run


class CompareGridRunsTests(unittest.TestCase):
    def test_summary_reports_throughput_memory_and_failures(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [
                {"started_at": 10.0, "finished_at": 12.0,
                 "timing": {"wall_seconds": 2.0},
                 "trainer_memory_after": {"peak_allocated_gb": 7.5}},
                {"started_at": 11.0, "finished_at": 13.0,
                 "timing": {"wall_seconds": 2.0},
                 "trainer_memory_after": {"peak_allocated_gb": 8.0}},
            ]
            (root / "episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
            (root / "failures.jsonl").write_text(json.dumps({"error": "planned"}) + "\n")
            summary = summarize_run(root)
            self.assertEqual(summary["episodes"], 2)
            self.assertEqual(summary["elapsed_seconds"], 3.0)
            self.assertAlmostEqual(summary["episodes_per_hour"], 2400.0)
            self.assertEqual(summary["trainer_peak_allocated_gb"], 8.0)
            self.assertEqual(summary["failure_attempts"], 1)


if __name__ == "__main__":
    unittest.main()
