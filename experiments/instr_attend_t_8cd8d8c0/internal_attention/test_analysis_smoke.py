#!/usr/bin/env python3
"""CPU smoke test for the offline aggregation contract."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

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
                (raw / f"{state}__{label}.tokens.json").write_text(
                    json.dumps({"instruction_indices": list(range(instruction_token_count))}),
                    encoding="utf-8",
                )
                rows.append({
                    "label": label,
                    "effect_size_vs_null": float(label_index),
                    "cosine": 1.0 - 0.01 * label_index,
                    "d_means": float(label_index),
                })
            attendance[state] = {"rows": rows}
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
    print("synthetic aggregation smoke: PASS")


if __name__ == "__main__":
    main()
