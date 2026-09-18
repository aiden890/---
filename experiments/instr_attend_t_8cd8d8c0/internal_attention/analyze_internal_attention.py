#!/usr/bin/env python3
"""Aggregate attention NPZ files, join action sensitivity, and render plots/report."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

STATES = ("reset", "move", "place")
LABELS = ("correct_full", "skill_grasp", "skill_move", "skill_place")


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    value = float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])
    return value if math.isfinite(value) else None


def attendance_lookup(path: Path) -> dict[tuple[str, str], dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        (state, row["label"]): row
        for state, state_data in raw.items()
        for row in state_data["rows"]
    }


def load_records(input_dir: Path, attendance_path: Path) -> tuple[list[dict], dict]:
    sensitivity = attendance_lookup(attendance_path)
    records = []
    for state in STATES:
        for label in LABELS:
            path = input_dir / f"{state}__{label}.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path) as data:
                arrays = {key: np.asarray(data[key]) for key in data.files}
            expected = {
                "vlm_instruction_to_image": (36, 32),
                "vlm_image_to_instruction": (36, 32),
                "dit_action_to_instruction": (5, 36, 8),
                "dit_action_to_image": (5, 36, 8),
                "dit_query_to_instruction": (5, 36, 8, 30),
            }
            for key, shape in expected.items():
                if arrays[key].shape != shape:
                    raise ValueError(f"{path.name}:{key} shape {arrays[key].shape} != {shape}")
                if not np.isfinite(arrays[key]).all():
                    raise ValueError(f"{path.name}:{key} contains non-finite values")
            action_instruction = arrays["dit_action_to_instruction"]
            action_image = arrays["dit_action_to_image"]
            sens = sensitivity[(state, label)]
            records.append({
                "state": state,
                "label": label,
                "npz": path.name,
                "action_to_instruction_mean": float(action_instruction.mean()),
                "action_to_instruction_std": float(action_instruction.std()),
                "action_to_image_mean": float(action_image.mean()),
                "instruction_to_image_mean": float(arrays["vlm_instruction_to_image"].mean()),
                "image_to_instruction_mean": float(arrays["vlm_image_to_instruction"].mean()),
                "GRASP_token_attention": float(arrays["dit_action_to_grasp_tokens"].mean()),
                "MOVE_token_attention": float(arrays["dit_action_to_move_tokens"].mean()),
                "PLACE_token_attention": float(arrays["dit_action_to_place_tokens"].mean()),
                "action_effect_size": float(sens["effect_size_vs_null"]),
                "action_cosine": float(sens["cosine"]),
                "action_d_means": float(sens["d_means"]),
                "arrays": arrays,
            })
    return records, sensitivity


def colorize(values: np.ndarray, low: float, high: float) -> Image.Image:
    scale = max(high - low, 1e-12)
    normalized = np.clip((values.astype(np.float32) - low) / scale, 0.0, 1.0)
    # Perceptually ordered blue→cyan→yellow ramp, implemented without a plotting dependency.
    red = np.clip(2.0 * normalized - 0.2, 0.0, 1.0)
    green = np.clip(2.0 * normalized, 0.0, 1.0)
    blue = np.clip(1.5 - 2.0 * normalized, 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(np.uint8(np.round(rgb * 255.0)), mode="RGB")


def panel_heatmaps(records: list[dict], array_key: str, reducer, path: Path, caption: str) -> None:
    panels = [reducer(record["arrays"][array_key]) for record in records]
    low = min(float(panel.min()) for panel in panels)
    high = max(float(panel.max()) for panel in panels)
    cell_w, cell_h, margin, title_h = 520, 260, 18, 42
    canvas = Image.new("RGB", (4 * cell_w, 3 * cell_h + 35), "white")
    draw = ImageDraw.Draw(canvas)
    for record, values in zip(records, panels):
        row = STATES.index(record["state"])
        col = LABELS.index(record["label"])
        heatmap = colorize(values, low, high).resize(
            (cell_w - 2 * margin, cell_h - title_h - margin), Image.Resampling.NEAREST
        )
        x, y = col * cell_w, row * cell_h
        canvas.paste(heatmap, (x + margin, y + title_h))
        draw.text((x + margin, y + 8), f"{record['state']} / {record['label']}", fill="black")
        draw.rectangle((x + margin, y + title_h, x + cell_w - margin, y + cell_h - margin), outline="black")
    draw.text((margin, 3 * cell_h + 8), f"{caption}; global range [{low:.6g}, {high:.6g}]", fill="black")
    canvas.save(path)


def plot_layer_timestep(records: list[dict], out: Path) -> None:
    panel_heatmaps(
        records,
        "dit_action_to_instruction",
        lambda values: values.mean(axis=2),
        out / "action_instruction_layer_timestep.png",
        "rows=flow timestep, columns=DiT layer",
    )


def plot_layer_head(records: list[dict], out: Path) -> None:
    panel_heatmaps(
        records,
        "dit_action_to_instruction",
        lambda values: values.mean(axis=0).T,
        out / "action_instruction_layer_head.png",
        "rows=head, columns=DiT layer; mean across flow steps",
    )


def plot_vlm(records: list[dict], out: Path) -> None:
    panel_heatmaps(
        records,
        "vlm_instruction_to_image",
        lambda values: values.T,
        out / "vlm_instruction_image_layer_head.png",
        "rows=head, columns=VLM layer",
    )


def plot_joint(records: list[dict], out: Path) -> None:
    width, height, margin = 1400, 900, 110
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    colors = {"reset": "blue", "move": "orange", "place": "green"}
    xs = [record["action_to_instruction_mean"] for record in records]
    ys = [record["action_effect_size"] for record in records]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xscale = max(xmax - xmin, 1e-12)
    yscale = max(ymax - ymin, 1e-12)
    draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
    draw.line((margin, margin, margin, height - margin), fill="black", width=2)
    for record in records:
        x = margin + (record["action_to_instruction_mean"] - xmin) / xscale * (width - 2 * margin)
        y = height - margin - (record["action_effect_size"] - ymin) / yscale * (height - 2 * margin)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=colors[record["state"]])
        draw.text((x + 8, y - 8), f"{record['state']}/{record['label'].replace('skill_', '')}", fill="black")
    draw.text((margin, height - 55), "mean DiT action->instruction attention mass", fill="black")
    draw.text((10, 20), "counterfactual action effect size", fill="black")
    draw.text((margin, height - margin + 12), f"x=[{xmin:.6g}, {xmax:.6g}]", fill="black")
    draw.text((10, height - margin - 18), f"y=[{ymin:.3g}, {ymax:.3g}]", fill="black")
    canvas.save(out / "attention_vs_action_sensitivity.png")


def write_csv(records: list[dict], out: Path) -> None:
    keys = [key for key in records[0] if key != "arrays"]
    with (out / "combined_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in keys})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--attendance", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records, _ = load_records(args.input, args.attendance)
    plot_layer_timestep(records, args.out)
    plot_layer_head(records, args.out)
    plot_vlm(records, args.out)
    plot_joint(records, args.out)
    write_csv(records, args.out)

    non_reference = [record for record in records if record["label"] != "correct_full"]
    all_corr = pearson(
        [record["action_to_instruction_mean"] for record in non_reference],
        [record["action_effect_size"] for record in non_reference],
    )
    by_state = {
        state: pearson(
            [r["action_to_instruction_mean"] for r in non_reference if r["state"] == state],
            [r["action_effect_size"] for r in non_reference if r["state"] == state],
        )
        for state in STATES
    }
    clean_records = [{key: value for key, value in record.items() if key != "arrays"} for record in records]
    provenance = json.loads((args.input / "server_provenance.json").read_text(encoding="utf-8"))
    summary = {
        "schema_version": 1,
        "cases": clean_records,
        "tensor_validation": {
            "case_count": len(records),
            "expected_case_count": 12,
            "vlm_shape": [36, 32],
            "dit_shape": [5, 36, 8],
            "dit_query_shape": [5, 36, 8, 30],
            "all_finite": True,
            "dtype": "float32",
        },
        "joint_analysis": {
            "pearson_attention_vs_effect_nonreference": all_corr,
            "pearson_by_state": by_state,
            "interpretation_rule": "correlation is descriptive cross-validation, not a causal estimate",
        },
        "provenance": provenance,
    }
    (args.out / "combined_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    state_lines = []
    for state in STATES:
        rows = [record for record in records if record["state"] == state]
        state_lines.append(
            f"| {state} | {np.mean([r['action_to_instruction_mean'] for r in rows]):.6f} | "
            f"{np.mean([r['action_to_image_mean'] for r in rows]):.6f} | "
            f"{np.mean([r['instruction_to_image_mean'] for r in rows]):.6f} | "
            f"{by_state[state] if by_state[state] is not None else 'N/A'} |"
        )
    report = f"""# 스킬별 내부 attention × 행동 민감도 결합 분석

## 검증 범위

- 실제 Xiaomi-Robotics-1-RoboCasa365 checkpoint forward의 Q/K projection과 RoPE를 그대로 사용했다.
- 정책 실행은 원래 FlashAttention/SDPA 경로를 유지했고, hook은 선택 query/key 확률만 read-only로 재구성했다.
- 12개 조건(reset/move/place × correct/grasp/move/place), VLM 36층×32헤드 및 DiT 5 flow timestep×36층×8헤드를 모두 기록했다.
- 원 tensor summary shape: VLM `(36, 32)`, DiT `(5, 36, 8)`, action-query 상세 `(5, 36, 8, 30)`; 저장 dtype float32.

## 상태별 요약

| state | action→instruction | action→image | instruction→image | attention↔행동 ES Pearson |
|---|---:|---:|---:|---:|
{chr(10).join(state_lines)}

비정답 skill 9개 조건 전체의 attention mass와 기존 action effect size 간 Pearson 상관은 `{all_corr}`이다. 이 값은 attention이 행동 민감도와 함께 움직이는지 보는 교차검증일 뿐 인과 추정치가 아니다.

## 해석

1. `action→instruction`은 DiT action query가 VLM KV cache의 instruction span에 둔 정규화 attention mass다. `action→image`와 동일 분모에서 측정했다.
2. `instruction→image`는 causal VLM에서 instruction query가 앞선 visual token에 둔 attention이다. 반대 방향 `image→instruction`은 image token이 instruction보다 먼저 오므로 causal mask상 정확히 0이다.
3. GRASP/MOVE/PLACE 열은 각 instruction 안에서 명시적으로 발견된 관련 lexeme token에 대한 mass다. 해당 lexeme가 없는 지시는 0으로 기록하며, token index는 각 `*.tokens.json`에 보존했다.
4. 높은 attention weight는 routing의 관찰값이지 causal importance가 아니다. 기존 고정-observation counterfactual action ES/cosine과 일치·불일치를 함께 봐야 한다.
5. token masking은 checkpoint forward를 바꾸므로 이번 read-only 측정에는 포함하지 않았다. 기존 instruction 교체 실험이 행동 수준의 독립적 intervention 역할을 한다.

## 산출물

- `combined_summary.json`, `combined_summary.csv`: machine-readable 결합 결과
- `action_instruction_layer_timestep.png`: timestep×layer heatmap
- `action_instruction_layer_head.png`: head×layer heatmap
- `vlm_instruction_image_layer_head.png`: VLM instruction→image heatmap
- `attention_vs_action_sensitivity.png`: attention–행동 ES 결합 scatter
- raw `*.npz`, `*.tokens.json`, `manifest.jsonl`: 조건별 tensor와 token mapping

## 한계

- attention probability는 value vector, residual stream, MLP, AdaLN gate의 영향을 포함하지 않으므로 단독으로 feature importance를 뜻하지 않는다.
- flow timestep별 attention은 같은 noisy action trajectory 안의 관찰이며 독립 표본이 아니다.
- 기존 행동 민감도는 instruction 교체 intervention이고 내부 attention run은 각 조건 1회이므로, 샘플링 노이즈까지 포함한 인과효과 크기로 읽으면 안 된다.
"""
    (args.out / "REPORT.ko.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary["tensor_validation"], ensure_ascii=False))


if __name__ == "__main__":
    main()
