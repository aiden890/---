#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


class ControlArtifactTest(unittest.TestCase):
    def test_registered_decision_branch(self) -> None:
        summary = json.loads((RESULTS / "summary.json").read_text())
        self.assertEqual(summary["verdict"], "checkpoint limitation")
        self.assertEqual(summary["xiaomi"]["general_qa_semantic_nonempty"], 0)
        self.assertEqual(summary["xiaomi"]["planning_semantic_nonempty"], 0)
        self.assertEqual(summary["official"]["semantic_nonempty"], 3)
        self.assertEqual(summary["official"]["direct_valid_json"], 3)

    def test_direct_generation_and_provenance(self) -> None:
        xiaomi = json.loads((RESULTS / "xiaomi_checkpoint_sanity.json").read_text())
        official = json.loads((RESULTS / "official_instruct.json").read_text())
        self.assertEqual(len(xiaomi["cases"]), 4)
        self.assertEqual(len(official["cases"]), 3)
        self.assertTrue(all(case["raw_output"] == "<cot></cot>" for case in xiaomi["cases"]))
        self.assertTrue(all(case["stop_reason"] == "eos_token" for case in xiaomi["cases"]))
        self.assertTrue(all(case["direct_valid_json"] for case in official["cases"]))
        self.assertEqual(len({case["image_sha256"] for case in official["cases"]}), 3)
        for case in xiaomi["cases"] + official["cases"]:
            self.assertTrue(case["generated_token_ids"])
            self.assertTrue(case["first_token_topk"])
            self.assertGreater(case["latency_ms"], 0)

    def test_no_repair_or_fallback(self) -> None:
        summary = json.loads((RESULTS / "summary.json").read_text())
        self.assertEqual(summary["constraints"], {
            "fallback_used": False,
            "parser_repair_used": False,
            "constrained_decoding_used": False,
            "production_runtime_modified": False,
        })


if __name__ == "__main__":
    unittest.main()
