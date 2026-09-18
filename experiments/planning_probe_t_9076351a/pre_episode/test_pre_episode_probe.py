#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = Path(__file__).with_name("pre_episode_probe.py")
CANONICAL_PATH = ROOT / "canonical_control" / "canonical_probe.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load("pre_episode_probe_test", PROBE_PATH)
canonical = load("canonical_probe_test", CANONICAL_PATH)
VALID = json.dumps({"plan": [
    {"name": "GRASP_OBJECT", "args": {"object": "blender_lid", "grasp_region": "lid_handle"}},
    {"name": "MOVE_OBJECT", "args": {"object": "blender_lid", "destination": "closed_preplace_region"}},
    {"name": "PLACE_OBJECT", "args": {"object": "blender_lid", "destination": "blender"}},
]})


class ContractTests(unittest.TestCase):
    def test_contract_constants(self) -> None:
        self.assertEqual(probe.POLICY_CHUNK_ACTIONS, 16)
        self.assertEqual(probe.ACTION_RATE_HZ, 20.0)
        self.assertEqual(probe.CHUNK_DEADLINE_MS, 800.0)
        self.assertEqual(probe.MAX_BUFFER_CHUNKS, 1)

    def test_valid_plan_is_preserved(self) -> None:
        plan, errors = probe.validate_raw_plan(VALID, canonical)
        self.assertEqual(errors, [])
        self.assertEqual([step["name"] for step in plan["plan"]], probe.EXPECTED_SEQUENCE)

    def test_every_invalid_mode_fails_closed_with_zero_actions(self) -> None:
        cases = probe.failure_mode_cases(VALID, canonical)
        self.assertEqual(len(cases), 6)
        self.assertTrue(all(case["pass"] for case in cases))
        self.assertTrue(all(case["actions_executed"] == 0 for case in cases))
        self.assertTrue(all(not case["fallback_used"] for case in cases))

    def test_wrong_sequence_is_not_corrected(self) -> None:
        raw = json.dumps({"plan": [
            {"name": "MOVE_OBJECT", "args": {"object": "blender_lid", "destination": "closed_preplace_region"}},
            {"name": "GRASP_OBJECT", "args": {"object": "blender_lid", "grasp_region": "lid_handle"}},
            {"name": "PLACE_OBJECT", "args": {"object": "blender_lid", "destination": "blender"}},
        ]})
        plan, errors = probe.validate_raw_plan(raw, canonical)
        self.assertIsNone(plan)
        self.assertIn("canonical sequence is not the required complete plan", errors)

    def test_percentile_and_distribution(self) -> None:
        self.assertEqual(probe.percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertAlmostEqual(probe.distribution([200.0, 250.0])["throughput_hz"], 1000.0 / 225.0)


if __name__ == "__main__":
    unittest.main()
