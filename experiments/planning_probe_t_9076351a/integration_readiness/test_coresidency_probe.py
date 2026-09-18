#!/usr/bin/env python3
"""CPU-only unit tests for co-residency probe aggregation helpers."""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")
if "PIL" not in sys.modules:
    pil = types.ModuleType("PIL")
    pil.Image = object
    sys.modules["PIL"] = pil
if "transformers" not in sys.modules:
    transformers = types.ModuleType("transformers")
    transformers.AutoModel = object
    transformers.AutoProcessor = object
    transformers.Qwen3VLForConditionalGeneration = object
    sys.modules["transformers"] = transformers

PATH = Path(__file__).with_name("coresidency_probe.py")
SPEC = importlib.util.spec_from_file_location("coresidency_probe", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AggregateTests(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        self.assertEqual(MODULE.percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertAlmostEqual(MODULE.percentile([1.0, 2.0], 0.95), 1.95)

    def test_distribution(self) -> None:
        result = MODULE.distribution([100.0, 200.0, 300.0])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["p50_ms"], 200.0)
        self.assertEqual(result["max_ms"], 300.0)
        self.assertAlmostEqual(result["throughput_hz"], 5.0)


if __name__ == "__main__":
    unittest.main()
