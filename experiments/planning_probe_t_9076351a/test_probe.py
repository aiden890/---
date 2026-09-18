#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

PATH = Path(__file__).with_name("probe.py")
SPEC = importlib.util.spec_from_file_location("planning_probe", PATH)
PROBE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PROBE)


class ValidationTest(unittest.TestCase):
    def call(self, name):
        spec = PROBE.BY_NAME[name]
        return {"name": name, "args": dict(spec["default_args"]),
                "instruction": spec["instruction"], "contract": spec["done_when"],
                "budget": spec["max_steps"]}

    def test_exact_plan(self):
        plan = {"plan": [self.call(name) for name in PROBE.EXPECTED]}
        scored = PROBE.score_plan(plan)
        self.assertTrue(scored["schema_valid"])
        self.assertTrue(scored["exact_sequence"])
        self.assertTrue(scored["first_skill_correct"])

    def test_rejects_wrong_field_and_budget(self):
        call = self.call("GRASP_OBJECT")
        call["budget"] = 999
        self.assertFalse(PROBE.validate_call(call)["budget_valid"])
        call = self.call("GRASP_OBJECT")
        call["extra"] = "not allowed"
        self.assertFalse(PROBE.validate_call(call)["schema_valid"])

    def test_rejects_noncanonical_args_and_contract(self):
        call = self.call("MOVE_OBJECT")
        call["args"]["destination"] = "blender"
        call["contract"] = "looks done"
        validity = PROBE.validate_call(call)
        self.assertFalse(validity["args_valid"])
        self.assertFalse(validity["contract_valid"])

    def test_detects_plan_failures(self):
        plan = {"plan": [self.call("GRASP_OBJECT"), self.call("GRASP_OBJECT"),
                         self.call("PLACE_OBJECT"), {"name": "CLOSE_LID"}]}
        scored = PROBE.score_plan(plan)
        self.assertTrue(scored["extra_step"])
        self.assertTrue(scored["duplicate_cycle"])
        self.assertTrue(scored["hallucinated_skill"])
        self.assertFalse(scored["exact_sequence"])

    def test_direct_and_repair_are_separate(self):
        raw = "```json\n" + json.dumps({"plan": []}) + "\n```"
        parsed, error = PROBE.direct_json(raw)
        self.assertIsNone(parsed)
        self.assertIsNotNone(error)
        self.assertEqual(PROBE.repaired_json(raw), {"plan": []})

    def test_prompts_do_not_leak_answer_or_gt(self):
        full = PROBE.full_plan_prompt(PROBE.INSTRUCTIONS[0])
        self.assertNotIn("GRASP_OBJECT → MOVE_OBJECT → PLACE_OBJECT", full)
        self.assertNotIn("official_check_success", full)
        self.assertNotIn("lid_grasped", full)


if __name__ == "__main__":
    unittest.main()
