from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grid_entry import parse_cli


class GridEntryTests(unittest.TestCase):
    def test_launcher_options_can_precede_hydra_overrides(self):
        args = parse_cli([
            "fake-smoke", "--config", str(ROOT / "configs" / "grid_smoke.yaml"),
            "--output", "/tmp/grid-test", "--mode", "serial", "--",
            "env.horizon=16", "rollout.replan_steps=8",
        ])
        self.assertEqual(args.mode, "serial")
        self.assertEqual(args.overrides, ["env.horizon=16", "rollout.replan_steps=8"])