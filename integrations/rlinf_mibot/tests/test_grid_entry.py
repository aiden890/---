from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grid_entry import load_config, parse_cli, partition_grid_overrides


class GridEntryTests(unittest.TestCase):
    def test_launcher_options_can_precede_hydra_overrides(self):
        args = parse_cli([
            "fake-smoke", "--config", str(ROOT / "configs" / "grid_smoke.yaml"),
            "--output", "/tmp/grid-test", "--mode", "serial", "--",
            "env.horizon=16", "rollout.replan_steps=8",
        ])
        self.assertEqual(args.mode, "serial")
        self.assertEqual(args.overrides, ["env.horizon=16", "rollout.replan_steps=8"])

    def test_dotted_grid_keys_are_not_misparsed_as_top_level_nested_overrides(self):
        grid, regular = partition_grid_overrides(
            {"env.horizon", "rollout.replan_steps"},
            ["env.horizon=16", "rollout.replan_steps=8", "validation.inject_fail_once_seed=1"],
        )
        self.assertEqual(grid, ["env.horizon=16", "rollout.replan_steps=8"])
        self.assertEqual(regular, ["validation.inject_fail_once_seed=1"])

    def test_optional_group_grid_keys_can_be_added_by_override(self):
        cfg = load_config(ROOT / "configs" / "grid_smoke.yaml",
                          ["rollout.groups=1", "rollout.group_size=4"])
        self.assertEqual(cfg["grid"]["rollout.groups"], 1)
        self.assertEqual(cfg["grid"]["rollout.group_size"], 4)
