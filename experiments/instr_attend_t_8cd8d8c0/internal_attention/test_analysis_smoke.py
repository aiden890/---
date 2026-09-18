#!/usr/bin/env python3
"""CPU smoke test for the offline aggregation contract."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

STATES = ("reset", "move", "place")
LABELS = ("correct_full", "skill_grasp", "skill_move", "skill_place")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    attendance = {}
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        raw = root / "raw"
        out = root / "out"
        raw.mkdir()
        for state_index, state in enumerate(STATES):
            rows = []
            for label_index, label in enumerate(LABELS):
                value = np.float32(0.001 * (1 + state_index + label_index))
                arrays = {
                    "vlm_instruction_to_image": np.full((36, 32), value, np.float32),
                    "vlm_image_to_instruction": np.zeros((36, 32), np.float32),
                    "vlm_instruction_self": np.full((36, 32), value, np.float32),
                    "dit_action_to_instruction": np.full((5, 36, 8), value, np.float32),
                    "dit_action_to_image": np.full((5, 36, 8), value * 2, np.float32),
                    "dit_action_to_text_total": np.full((5, 36, 8), value * 3, np.float32),
                    "dit_query_to_instruction": np.full((5, 36, 8, 16), value, np.float32),
                    "dit_action_to_grasp_tokens": np.full((5, 36, 8), value, np.float32),
                    "dit_action_to_move_tokens": np.full((5, 36, 8), value, np.float32),
                    "dit_action_to_place_tokens": np.full((5, 36, 8), value, np.float32),
                    "actions": np.zeros((1, 16, 60), np.float32),
                }
                np.savez_compressed(raw / f"{state}__{label}.npz", **arrays)
                instruction_token_count = (12, 7, 16, 23)[label_index]
                instruction_indices = list(range(10, 10 + instruction_token_count))
                semantic = {
                    "GRASP": [instruction_indices[0]] if label == "skill_grasp" else [],
                    "MOVE": [instruction_indices[0]] if label == "skill_move" else [],
                    "PLACE": [instruction_indices[0]] if label in ("correct_full", "skill_place") else [],
                }
                tokens = [
                    {"index": 0, "token": "<|im_start|>"},
                    {"index": 2, "token": "<|vision_start|>"},
                    {"index": 3, "token": "<|video_pad|>"},
                    {"index": 4, "token": "<|vision_end|>"},
                    {"index": 10 + instruction_token_count, "token": "<|im_end|>"},
                ]
                (raw / f"{state}__{label}.tokens.json").write_text(
                    json.dumps({
                        "meta": {"instruction": label, "num_steps": 5, "action_length": 16},
                        "sequence_length": 11 + instruction_token_count,
                        "instruction_indices": instruction_indices,
                        "image_indices": [3],
                        "semantic_indices": semantic,
                        "tokens": tokens,
                    }),
                    encoding="utf-8",
                )
                rows.append({
                    "label": label,
                    "effect_size_vs_null": float(label_index),
                    "cosine": 1.0 - 0.01 * label_index,
                    "d_means": float(label_index),
                })
            attendance[state] = {
                "N": 24,
                "noise_floor_pairwise_L2": 1.0,
                "null_meandiff_scale": 1.0 / (24 ** 0.5),
                "rows": rows,
            }
        (raw / "server_provenance.json").write_text(
            json.dumps({"gpu": "synthetic", "checkpoint": "synthetic"}), encoding="utf-8"
        )
        attendance_path = root / "attendance.json"
        attendance_path.write_text(json.dumps(attendance), encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                str(script_dir / "analyze_internal_attention.py"),
                "--input", str(raw),
                "--attendance", str(attendance_path),
                "--out", str(out),
            ],
            check=True,
        )
        summary = json.loads((out / "combined_summary.json").read_text(encoding="utf-8"))
        assert summary["tensor_validation"]["case_count"] == 12
        assert summary["tensor_validation"]["all_finite"] is True
        assert summary["schema_version"] == 2
        guide = summary["interpretation_guide"]
        assert guide["direction_and_normalization"]["direction"] == "query -> key"
        assert guide["aggregation"]["attention_samples_per_condition"] == 1
        assert guide["token_boundaries"]["skill_place"]["action_query_index_range"] == [0, 15]
        assert "attention != causal importance" in guide["caveat"]
        expected = {
            "combined_summary.csv",
            "REPORT.ko.md",
            "action_instruction_layer_timestep.png",
            "action_instruction_layer_head.png",
            "vlm_instruction_image_layer_head.png",
            "attention_vs_action_sensitivity.png",
        }
        missing = sorted(name for name in expected if not (out / name).exists())
        assert not missing, missing
        for name in expected:
            if name.endswith(".png"):
                with Image.open(out / name) as image:
                    assert image.width >= 1800 and image.height >= 800, (name, image.size)
        report = (out / "REPORT.ko.md").read_text(encoding="utf-8")
        for required in ("## 먼저 읽는 법", "## 축과 tensor 의미", "## 집계와 정규화", "## token 경계", "## 한계"):
            assert required in report, required
    print("synthetic aggregation smoke: PASS")


if __name__ == "__main__":
    main()
