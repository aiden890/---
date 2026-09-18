#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("canonical_probe", ROOT / "canonical_probe.py")
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)
RESULTS = ROOT / "results"


class CanonicalizerUnitTest(unittest.TestCase):
    def good_plan(self):
        return {"plan": [
            {"name": "GRASP_OBJECT", "args": {"object": "blender_lid", "grasp_region": "lid_handle"}},
            {"name": "MOVE_OBJECT", "args": {"object": "blender_lid", "destination": "closed_preplace_region"}},
            {"name": "PLACE_OBJECT", "args": {"object": "blender_lid", "destination": "blender"}},
        ]}

    def test_materializes_strict_fields_without_changing_semantics(self):
        source = self.good_plan()
        canonical, errors = PROBE.canonicalize(source)
        self.assertEqual(errors, [])
        self.assertTrue(PROBE.strict_plan_valid(canonical))
        self.assertEqual([step["name"] for step in canonical["plan"]], [step["name"] for step in source["plan"]])
        self.assertEqual([step["args"] for step in canonical["plan"]], [step["args"] for step in source["plan"]])

    def test_rejects_unknown_missing_extra_duplicate_and_disallowed(self):
        bad = []
        unknown = self.good_plan(); unknown["plan"][0]["name"] = "MAGIC"
        bad.append(unknown)
        missing = self.good_plan(); missing["plan"].pop()
        bad.append(missing)
        extra = self.good_plan(); extra["plan"].append(dict(extra["plan"][-1]))
        bad.append(extra)
        duplicate = self.good_plan(); duplicate["plan"][1] = dict(duplicate["plan"][0])
        bad.append(duplicate)
        value = self.good_plan(); value["plan"][0]["args"]["grasp_region"] = "handle"
        bad.append(value)
        fields = self.good_plan(); fields["plan"][0]["budget"] = 208
        bad.append(fields)
        for plan in bad:
            canonical, errors = PROBE.canonicalize(plan)
            self.assertIsNone(canonical)
            self.assertTrue(errors)


@unittest.skipUnless((RESULTS / "summary.json").exists(), "generation results not present")
class GeneratedArtifactTest(unittest.TestCase):
    def test_hard_gate_and_case_counts(self):
        summary = json.loads((RESULTS / "summary.json").read_text())
        self.assertEqual(summary["cases"], 12)
        self.assertEqual(summary["verdict"], "PASS")
        for key in (
            "direct_semantic_json", "exact_three_skill_sequence",
            "deterministic_canonicalization_success", "canonicalized_strict_five_field_plan",
        ):
            self.assertEqual(summary[key], 12)

    def test_no_hidden_repair_and_separate_outputs(self):
        data = json.loads((RESULTS / "cases.json").read_text())
        self.assertEqual(len(data["cases"]), 12)
        self.assertEqual({case["seed"] for case in data["cases"]}, {0, 1, 2})
        self.assertEqual({case["instruction_index"] for case in data["cases"]}, {0, 1, 2, 3})
        self.assertEqual(len({case["image_sha256"] for case in data["cases"]}), 3)
        self.assertEqual(len({case["instruction"] for case in data["cases"]}), 4)
        self.assertEqual(data["constraints"], {
            "fallback_used": False,
            "parser_repair_used": False,
            "constrained_decoding_used": False,
            "production_runtime_modified": False,
        })
        for case in data["cases"]:
            self.assertTrue((RESULTS / case["raw_path"]).is_file())
            self.assertTrue((RESULTS / case["canonical_path"]).is_file())
            self.assertTrue(PROBE.strict_plan_valid(case["canonicalized_plan"]))


if __name__ == "__main__":
    unittest.main()
