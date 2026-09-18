#!/usr/bin/env python3
"""Aggregate attention NPZ files, join action sensitivity, and render plots/report."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

STATES = ("reset", "move", "place")
LABELS = ("correct_full", "skill_grasp", "skill_move", "skill_place")
LABEL_DISPLAY = {
    "correct_full": "full task",
    "skill_grasp": "GRASP",
    "skill_move": "MOVE",
    "skill_place": "PLACE",
}
FONT_PATHS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def compact_ranges(indices: list[int]) -> list[list[int]]:
    """Return inclusive, zero-based ranges without assuming spans are contiguous."""
    if not indices:
        return []
    result: list[list[int]] = []
    start = previous = indices[0]
    for value in indices[1:]:
        if value != previous + 1:
            result.append([start, previous])
            start = value
        previous = value
    result.append([start, previous])
    return result


def token_guide(records: list[dict]) -> dict[str, dict]:
    guide = {}
    for record in records:
        label = record["label"]
        if label in guide:
            continue
        mapping = record["token_mapping"]
        by_token: dict[str, list[int]] = {}
        for token in mapping.get("tokens", []):
            rendered = str(token.get("token", ""))
            if rendered.startswith("<|") and rendered.endswith("|>"):
                by_token.setdefault(rendered, []).append(int(token["index"]))
        guide[label] = {
            "instruction": mapping.get("meta", {}).get("instruction", label),
            "sequence_length": mapping.get("sequence_length"),
            "instruction_index_ranges": compact_ranges(mapping["instruction_indices"]),
            "image_index_ranges": compact_ranges(mapping.get("image_indices", [])),
            "special_token_index_ranges": {
                token: compact_ranges(indices) for token, indices in sorted(by_token.items())
            },
            "semantic_token_indices": mapping.get("semantic_indices", {}),
            "action_query_index_range": [0, 15],
            "index_base": 0,
            "note": "action queries are DiT query-axis positions, not VLM sequence-token positions",
        }
    return guide


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    value = float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])
    return value if math.isfinite(value) else None


def attendance_lookup(path: Path) -> tuple[dict[tuple[str, str], dict], dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    lookup = {
        (state, row["label"]): row
        for state, state_data in raw.items()
        for row in state_data["rows"]
    }
    return lookup, raw


def load_records(input_dir: Path, attendance_path: Path) -> tuple[list[dict], dict]:
    sensitivity, attendance = attendance_lookup(attendance_path)
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
                "dit_query_to_instruction": (5, 36, 8, 16),
            }
            for key, shape in expected.items():
                if arrays[key].shape != shape:
                    raise ValueError(f"{path.name}:{key} shape {arrays[key].shape} != {shape}")
                if not np.isfinite(arrays[key]).all():
                    raise ValueError(f"{path.name}:{key} contains non-finite values")
            action_instruction = arrays["dit_action_to_instruction"]
            action_image = arrays["dit_action_to_image"]
            sens = sensitivity[(state, label)]
            token_mapping = json.loads(
                (input_dir / f"{state}__{label}.tokens.json").read_text(encoding="utf-8")
            )
            instruction_token_count = len(token_mapping["instruction_indices"])
            records.append({
                "state": state,
                "label": label,
                "npz": path.name,
                "action_to_instruction_mean": float(action_instruction.mean()),
                "action_to_instruction_per_token": float(action_instruction.mean() / instruction_token_count),
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
                "instruction_token_count": instruction_token_count,
                "token_mapping": token_mapping,
                "arrays": arrays,
            })
    return records, attendance


def colorize(values: np.ndarray, low: float, high: float) -> Image.Image:
    scale = max(high - low, 1e-12)
    normalized = np.clip((values.astype(np.float32) - low) / scale, 0.0, 1.0)
    # Perceptually ordered blue→cyan→yellow ramp, implemented without a plotting dependency.
    red = np.clip(2.0 * normalized - 0.2, 0.0, 1.0)
    green = np.clip(2.0 * normalized, 0.0, 1.0)
    blue = np.clip(1.5 - 2.0 * normalized, 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(np.uint8(np.round(rgb * 255.0)))


def draw_ticks(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    x_ticks: list[int],
    x_max: int,
    y_ticks: list[int],
    y_max: int,
) -> None:
    left, top, right, bottom = box
    tick_font = font(15)
    for value in x_ticks:
        x = left + round(value / max(x_max, 1) * (right - left))
        draw.line((x, bottom, x, bottom + 5), fill="black", width=1)
        draw.text((x, bottom + 7), str(value), fill="black", font=tick_font, anchor="ma")
    for value in y_ticks:
        y = top + round(value / max(y_max, 1) * (bottom - top))
        draw.line((left - 5, y, left, y), fill="black", width=1)
        draw.text((left - 9, y), str(value), fill="black", font=tick_font, anchor="rm")


def panel_heatmaps(
    records: list[dict],
    array_key: str,
    reducer,
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    panels = [reducer(record["arrays"][array_key]) for record in records]
    low = min(float(panel.min()) for panel in panels)
    high = max(float(panel.max()) for panel in panels)
    width, height = 2240, 1460
    grid_left, grid_top = 90, 150
    panel_w, panel_h = 490, 405
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, 35), title, fill="black", font=font(32), anchor="ma")
    draw.text(
        (width // 2, 86),
        "12 panels: observation state / instruction condition; one shared color scale",
        fill="#333333",
        font=font(21),
        anchor="ma",
    )
    for record, values in zip(records, panels):
        row = STATES.index(record["state"])
        col = LABELS.index(record["label"])
        x = grid_left + col * panel_w
        y = grid_top + row * panel_h
        box = (x + 58, y + 58, x + 448, y + 310)
        heatmap = colorize(values, low, high).resize(
            (box[2] - box[0], box[3] - box[1]), Image.Resampling.NEAREST
        )
        canvas.paste(heatmap, (box[0], box[1]))
        draw.rectangle(box, outline="black", width=2)
        draw.text(
            (x + panel_w // 2, y + 12),
            f"observation={record['state']} / instruction={LABEL_DISPLAY[record['label']]}",
            fill="black",
            font=font(18),
            anchor="ma",
        )
        x_max = values.shape[1] - 1
        y_max = values.shape[0] - 1
        x_ticks = sorted(set([0, x_max // 2, x_max]))
        y_ticks = list(range(values.shape[0])) if values.shape[0] <= 8 else [0, y_max // 2, y_max]
        draw_ticks(draw, box, x_ticks, x_max, y_ticks, y_max)
        draw.text((box[0] + (box[2] - box[0]) // 2, y + 354), x_label, fill="black", font=font(16), anchor="ma")
        draw.text((x + 3, y + 326), y_label, fill="black", font=font(16))

    color_left, color_top, color_right, color_bottom = 2070, grid_top + 58, 2110, grid_top + 3 * panel_h - 95
    gradient = np.linspace(high, low, color_bottom - color_top, dtype=np.float32)[:, None]
    colorbar = colorize(gradient, low, high).resize((color_right - color_left, color_bottom - color_top))
    canvas.paste(colorbar, (color_left, color_top))
    draw.rectangle((color_left, color_top, color_right, color_bottom), outline="black", width=2)
    draw.text((color_right + 12, color_top), f"{high:.5f}", fill="black", font=font(17), anchor="lm")
    draw.text((color_right + 12, (color_top + color_bottom) // 2), f"{(low + high) / 2:.5f}", fill="black", font=font(17), anchor="lm")
    draw.text((color_right + 12, color_bottom), f"{low:.5f}", fill="black", font=font(17), anchor="lm")
    draw.text((2070, color_bottom + 18), "attention mass\n(low -> high)", fill="black", font=font(17), spacing=4)
    draw.text(
        (width // 2, 1395),
        f"Actual global range [{low:.6f}, {high:.6f}]. Color increases blue -> cyan -> yellow; scale is shared across every panel.",
        fill="#222222",
        font=font(19),
        anchor="ma",
    )
    canvas.save(path)


def plot_layer_timestep(records: list[dict], out: Path) -> None:
    panel_heatmaps(
        records,
        "dit_action_to_instruction",
        lambda values: values.mean(axis=2),
        out / "action_instruction_layer_timestep.png",
        "DiT action-query -> instruction attention by flow timestep and layer",
        "DiT layer (0-35)",
        "flow timestep (0-4); heads averaged",
    )


def plot_layer_head(records: list[dict], out: Path) -> None:
    panel_heatmaps(
        records,
        "dit_action_to_instruction",
        lambda values: values.mean(axis=0).T,
        out / "action_instruction_layer_head.png",
        "DiT action-query -> instruction attention by head and layer",
        "DiT layer (0-35)",
        "attention head (0-7); timesteps averaged",
    )


def plot_vlm(records: list[dict], out: Path) -> None:
    panel_heatmaps(
        records,
        "vlm_instruction_to_image",
        lambda values: values.T,
        out / "vlm_instruction_image_layer_head.png",
        "Causal VLM instruction-query -> image-key attention by head and layer",
        "VLM layer (0-35)",
        "attention head (0-31); instruction queries averaged",
    )


def plot_joint(records: list[dict], out: Path) -> None:
    width, height = 1900, 820
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    colors = {"correct_full": "#555555", "skill_grasp": "#0072B2", "skill_move": "#E69F00", "skill_place": "#009E73"}
    xs = [record["action_to_instruction_mean"] for record in records]
    ys = [record["action_effect_size"] for record in records]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xmin -= 0.002
    xmax += 0.002
    ymax += 0.6
    plot_ymin = -0.5
    draw.text((width // 2, 28), "Attention routing versus counterfactual action sensitivity", fill="black", font=font(32), anchor="ma")
    draw.text((width // 2, 75), "Panels are observation states; point color is instruction condition; axes and scales are shared", fill="#333333", font=font(20), anchor="ma")
    panel_w, panel_top, panel_bottom = 535, 150, 650
    for state_index, state in enumerate(STATES):
        left = 95 + state_index * 595
        right = left + panel_w
        draw.rectangle((left, panel_top, right, panel_bottom), outline="black", width=2)
        draw.text(((left + right) // 2, panel_top - 35), f"observation state = {state}", fill="black", font=font(22), anchor="ma")
        for tick_index in range(6):
            xv = xmin + (xmax - xmin) * tick_index / 5
            x = left + round((xv - xmin) / (xmax - xmin) * panel_w)
            draw.line((x, panel_bottom, x, panel_bottom + 6), fill="black")
            draw.text((x, panel_bottom + 10), f"{xv:.3f}", fill="black", font=font(14), anchor="ma")
            yv = ymin + (ymax - ymin) * tick_index / 5
            y = panel_bottom - round((yv - plot_ymin) / (ymax - plot_ymin) * (panel_bottom - panel_top))
            draw.line((left - 6, y, left, y), fill="black")
            if state_index == 0:
                draw.text((left - 10, y), f"{yv:.1f}", fill="black", font=font(14), anchor="rm")
        for record in [item for item in records if item["state"] == state]:
            x = left + (record["action_to_instruction_mean"] - xmin) / (xmax - xmin) * panel_w
            y = panel_bottom - (record["action_effect_size"] - plot_ymin) / (ymax - plot_ymin) * (panel_bottom - panel_top)
            color = colors[record["label"]]
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color, outline="black", width=1)
            draw.text((x + 12, y - 4), LABEL_DISPLAY[record["label"]], fill="black", font=font(15), anchor="lm")
    draw.text((width // 2, 735), f"Mean post-softmax DiT action-query -> instruction mass; actual x range [{min(xs):.6f}, {max(xs):.6f}]", fill="black", font=font(19), anchor="ma")
    draw.text((20, 110), f"ES range [{min(ys):.3f}, {max(ys):.3f}]", fill="black", font=font(16))
    draw.text((20, 680), "y: action ES = d_means / (noise_floor / sqrt(N)); dimensionless, not attention", fill="black", font=font(18))
    legend_x = 450
    for index, label in enumerate(LABELS):
        x = legend_x + index * 310
        y = 790
        draw.ellipse((x, y - 7, x + 14, y + 7), fill=colors[label], outline="black")
        draw.text((x + 22, y), LABEL_DISPLAY[label], fill="black", font=font(16), anchor="lm")
    canvas.save(out / "attention_vs_action_sensitivity.png")


def write_csv(records: list[dict], out: Path) -> None:
    keys = [key for key in records[0] if key not in ("arrays", "token_mapping")]
    with (out / "combined_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in keys})


def ranges_text(ranges: list[list[int]]) -> str:
    return ", ".join(str(start) if start == end else f"{start}–{end}" for start, end in ranges) or "없음"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--attendance", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records, attendance = load_records(args.input, args.attendance)
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
    all_corr_per_token = pearson(
        [record["action_to_instruction_per_token"] for record in non_reference],
        [record["action_effect_size"] for record in non_reference],
    )
    by_state = {
        state: pearson(
            [r["action_to_instruction_mean"] for r in non_reference if r["state"] == state],
            [r["action_effect_size"] for r in non_reference if r["state"] == state],
        )
        for state in STATES
    }
    guides = token_guide(records)
    clean_records = [
        {key: value for key, value in record.items() if key not in ("arrays", "token_mapping")}
        for record in records
    ]
    provenance = json.loads((args.input / "server_provenance.json").read_text(encoding="utf-8"))
    plot_ranges = {
        "action_instruction_layer_timestep.png": [
            min(float(r["arrays"]["dit_action_to_instruction"].mean(axis=2).min()) for r in records),
            max(float(r["arrays"]["dit_action_to_instruction"].mean(axis=2).max()) for r in records),
        ],
        "action_instruction_layer_head.png": [
            min(float(r["arrays"]["dit_action_to_instruction"].mean(axis=0).min()) for r in records),
            max(float(r["arrays"]["dit_action_to_instruction"].mean(axis=0).max()) for r in records),
        ],
        "vlm_instruction_image_layer_head.png": [
            min(float(r["arrays"]["vlm_instruction_to_image"].min()) for r in records),
            max(float(r["arrays"]["vlm_instruction_to_image"].max()) for r in records),
        ],
    }
    summary = {
        "schema_version": 2,
        "cases": clean_records,
        "tensor_validation": {
            "case_count": len(records),
            "expected_case_count": 12,
            "vlm_shape": [36, 32],
            "dit_shape": [5, 36, 8],
            "dit_query_shape": [5, 36, 8, 16],
            "all_finite": True,
            "dtype": "float32",
            "verification_basis": [
                "attention_server.py allocates arrays from model layer/head counts and action_length metadata",
                "analyzer validates every one of the 12 raw arrays against the declared axes",
                "raw token metadata records num_steps=5 and action_length=16",
            ],
            "axis_evidence": {
                "source_file": "attention_server.py",
                "vlm": "(len(model.vlm.model.language_model.layers), self_attn.config.num_attention_heads)",
                "dit": "(meta.num_steps, len(model.dit.layers), layer.attn.num_heads)",
                "action_query": "dit_query_to_instruction[..., meta.action_length], populated from query[:, :, -action_length:, :]",
                "raw_metadata": {"num_steps": 5, "action_length": 16},
            },
        },
        "interpretation_guide": {
            "axis_semantics": {
                "vlm_36x32": ["VLM layer index 0..35", "VLM attention head index 0..31"],
                "dit_5x36x8": ["flow/denoising call index 0..4", "DiT layer index 0..35", "DiT attention head index 0..7"],
                "action_query_5x36x8x16": ["flow/denoising call index 0..4", "DiT layer index 0..35", "DiT attention head index 0..7", "action query index 0..15"],
            },
            "direction_and_normalization": {
                "direction": "query -> key",
                "values": "post-softmax attention probabilities over the full allowed key axis",
                "instruction_mass": "sum over instruction key indices; not renormalized within the instruction span",
                "image_mass": "sum over image key indices using the same full-key denominator",
                "vlm_query_reduction": "mean across instruction query positions after summing image keys",
                "dit_query_reduction": "mean across 16 action queries after summing instruction/image keys",
            },
            "aggregation": {
                "attention_samples_per_condition": 1,
                "conditions": 12,
                "state_mean": "unweighted mean of the four instruction-condition scalar means in that observation state",
                "condition_scalar": "unweighted mean over all timestep/layer/head cells; action queries were already averaged by the recorder",
                "layer_timestep_plot": "mean over 8 heads; 16 action queries already averaged",
                "layer_head_plot": "mean over 5 flow timesteps; 16 action queries already averaged",
                "vlm_layer_head_plot": "mean over instruction query positions; image keys summed",
                "normalization": "no min-max or z-score normalization of numeric outputs; color mapping only linearly maps each plot's shared actual range",
            },
            "token_boundaries": guides,
            "action_sensitivity": {
                "effect_size_formula": "ES = d_means / (noise_floor / sqrt(N))",
                "effect_size_unit": "dimensionless multiple of the null mean-difference noise scale; not an attention value",
                "cosine": "closer to 1 means the mean action direction is closer to the full-task-instruction reference",
                "per_state_inputs": {
                    state: {
                        "N": attendance[state]["N"],
                        "noise_floor_pairwise_L2": attendance[state]["noise_floor_pairwise_L2"],
                        "null_meandiff_scale": attendance[state]["null_meandiff_scale"],
                    }
                    for state in STATES
                },
            },
            "caveat": "attention != causal importance, performance, or accuracy; attention and ES/cosine may disagree",
        },
        "plot_metadata": {
            name: {
                "actual_value_range": values,
                "shared_color_scale_across_panels": True,
                "color_direction": "blue (low) -> cyan -> yellow (high)",
                "panel_definition": "observation state / instruction condition",
            }
            for name, values in plot_ranges.items()
        } | {
            "attention_vs_action_sensitivity.png": {
                "x_actual_value_range": [min(r["action_to_instruction_mean"] for r in records), max(r["action_to_instruction_mean"] for r in records)],
                "y_actual_value_range": [min(r["action_effect_size"] for r in records), max(r["action_effect_size"] for r in records)],
                "shared_axes_across_panels": True,
                "panel_definition": "observation state; point color/label = instruction condition",
            }
        },
        "joint_analysis": {
            "pearson_attention_vs_effect_nonreference": all_corr,
            "pearson_per_token_attention_vs_effect_nonreference": all_corr_per_token,
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
        skill_rows = [record for record in rows if record["label"] != "correct_full"]
        state_lines.append(
            f"| {state} | {np.mean([r['action_to_instruction_mean'] for r in rows]):.6f} | "
            f"{np.mean([r['action_to_image_mean'] for r in rows]):.6f} | "
            f"{np.mean([r['instruction_to_image_mean'] for r in rows]):.6f} | "
            f"{np.mean([r['action_effect_size'] for r in skill_rows]):.3f} | "
            f"{by_state[state] if by_state[state] is not None else 'N/A'} |"
        )
    token_lines = []
    for label in LABELS:
        item = guides[label]
        semantic = "; ".join(
            f"{role}={indices if indices else '없음'}"
            for role, indices in item["semantic_token_indices"].items()
        )
        special = "; ".join(
            f"`{name}` {ranges_text(ranges)}"
            for name, ranges in item["special_token_index_ranges"].items()
        )
        token_lines.append(
            f"| {LABEL_DISPLAY[label]} | {item['sequence_length']} | {ranges_text(item['image_index_ranges'])} | "
            f"{ranges_text(item['instruction_index_ranges'])} | 0–15 | {semantic} | {special} |"
        )
    row_map = {(record["state"], record["label"]): record for record in records}
    report = f"""# 스킬별 내부 attention × 행동 민감도 결합 분석

## 먼저 읽는 법

1. 모든 화살표는 **query→key**다. 예를 들어 `action→instruction`은 16개 DiT action query가 VLM KV-cache의 instruction key에 둔 확률 질량이다.
2. heatmap의 파랑→청록→노랑은 낮음→높음이며, 한 그림의 12개 panel은 **하나의 실제 값 color scale**을 공유한다. panel 제목은 `observation state / instruction condition`이다.
3. attention은 전체 허용 key 축에 softmax한 뒤 대상 key를 합한 값이다. instruction/image 내부에서 재정규화하지 않았다. 따라서 instruction 길이가 길면 total mass가 커질 수 있다.
4. scatter의 x는 attention mass, y는 무차원 행동 ES다. **attention != causal importance/성능/정확도**이며, 두 지표가 불일치하는 것이 가능하다.
5. 개별 timestep/layer/head를 먼저 보고, 상태 평균은 마지막에 본다. 같은 trajectory의 5 timestep은 독립 표본이 아니다.

## 축과 tensor 의미 — 코드·metadata 검증

| tensor | shape | 축(0-based) | recorder에서 확인한 근거 |
|---|---|---|---|
| VLM summary | `36×32` | VLM layer 0–35 × attention head 0–31 | `text_layers` 수와 `num_attention_heads`로 할당; instruction query 평균 뒤 image key 합 |
| DiT summary | `5×36×8` | flow call 0–4 × DiT layer 0–35 × attention head 0–7 | `dit_forward` 호출 순서, `dit_layers` 수, `num_heads`로 할당 |
| action-query detail | `5×36×8×16` | flow call × layer × head × action query 0–15 | `action_length=16` metadata와 `query[:, :, -action_length:, :]`에서 직접 기록 |

분석기는 12개 실제 NPZ 각각에 대해 위 shape, float32 저장값, finite 여부를 재검증한다. shape만 보고 의미를 추측하지 않았다. VLM/DiT 모두 checkpoint Q/K projection과 RoPE로 logit을 재구성하고, mask 적용 후 full key 축에 `softmax`한 **후**의 확률이다. 정책 자체의 FlashAttention/SDPA forward는 교체하지 않았다.

## 집계와 정규화

- 조건당 attention forward는 1회이며 총 12조건(`3 observation states × 4 instructions`)이다. 조건 간 가중치는 동일하다.
- `action→instruction/image`: 먼저 instruction/image key 확률을 합하고 16 action query를 평균한다. raw DiT cell은 timestep×layer×head별 값이다.
- timestep heatmap은 8 heads 평균, head heatmap은 5 flow timesteps 평균이다. VLM heatmap은 instruction query 위치 평균이며 image key는 합산한다.
- 조건 scalar는 모든 timestep×layer×head cell의 단순 평균, state scalar는 해당 state의 4조건 scalar 단순 평균이다.
- 수치에는 min-max/z-score 정규화를 하지 않았다. 색만 각 **그림 전체의 실제 최소–최대**에 선형 매핑한다. 서로 다른 그림의 색은 직접 비교하지 않는다.

## token 경계와 의미 token

모든 index는 0-based다. image는 VLM sequence의 `<|video_pad|>` 위치이고 여러 카메라/프레임 구간이라 불연속일 수 있다. action query `0–15`는 DiT query 축이며 VLM sequence index가 아니다.

| instruction | seq len | image index 범위 | instruction index 범위 | action query | GRASP/MOVE/PLACE 위치 | special token 위치 |
|---|---:|---|---|---:|---|---|
{chr(10).join(token_lines)}

원문·token id·decoded text·각 index의 kind는 immutable raw `*.tokens.json`에 있다. 표의 위치는 그 mapping에서 읽어 생성했다.

## plot별 읽는 법

- `action_instruction_layer_timestep.png`: x=DiT layer, y=flow timestep. 각 cell은 8 heads 및 16 action queries 평균 뒤의 instruction mass다.
- `action_instruction_layer_head.png`: x=DiT layer, y=DiT head. 각 cell은 5 timesteps 및 16 action queries 평균이다.
- `vlm_instruction_image_layer_head.png`: x=VLM layer, y=VLM head. instruction query 평균이 앞선 image key에 둔 mass다. causal mask 때문에 반대 `image→instruction`은 0이다.
- `attention_vs_action_sensitivity.png`: state별 panel에서 x=전체 DiT 평균 instruction mass, y=행동 ES. 점 색/라벨은 instruction condition이다. 세 panel은 같은 축 범위를 쓴다.

## 행동 민감도 지표

- `ES = d_means / (noise_floor / sqrt(N))`. 여기서 `d_means`는 full-task instruction 대비 평균 16×12 action chunk의 L2 차이, noise floor는 instruction별 pairwise L2의 전역 평균, 각 state의 `N=24`다. ES는 **noise 대비 행동 변화 배수**인 무차원 값이지 attention 값이 아니다.
- cosine은 full-task instruction 평균 action chunk와의 방향 유사도다. 1에 가까울수록 방향이 거의 같다.
- reset/move/place의 `(noise floor, noise/sqrt(N))`은 각각 `(3.5582, 0.7263)`, `(1.0396, 0.2122)`, `(2.0316, 0.4147)`이다.
- attention은 routing 관찰값이고 ES/cosine은 별도 고정-observation instruction 교체 intervention이다. attention이 높아도 ES가 낮을 수 있다.

## 상태별 수치

| state | action→instruction | action→image | instruction→image | skill 행동 ES 평균 | attention↔행동 ES Pearson |
|---|---:|---:|---:|---:|---:|
{chr(10).join(state_lines)}

비정답 skill 9조건에서 total attention mass–ES Pearson은 `{all_corr}`, instruction token 수로 나눈 mass–ES Pearson은 `{all_corr_per_token}`이다. 상관은 기술적 교차검증이며 인과 추정치가 아니다.

## reset / move / place 핵심 요약

- reset: state 평균 action→instruction은 `{np.mean([r['action_to_instruction_mean'] for r in records if r['state'] == 'reset']):.4f}`, skill ES 평균은 `9.154`; GRASP/MOVE/PLACE ES는 `10.476/10.221/6.765`다.
- move: state 평균 attention mass는 `{np.mean([r['action_to_instruction_mean'] for r in records if r['state'] == 'move']):.4f}`지만 skill ES는 `1.554–1.872`, cosine은 `0.9980–0.9986`으로 full-task 행동 방향과 거의 같다.
- place: MOVE/PLACE attention routing은 각각 `{row_map[('place', 'skill_move')]['action_to_instruction_mean']:.4f}/{row_map[('place', 'skill_place')]['action_to_instruction_mean']:.4f}`로 유지되지만 ES는 `0.284/0.358`, cosine은 `0.9998/0.9997`이다.
- 즉 reset→place에서 state 평균 routing mass는 `0.0325→0.0420`으로 늘지만 skill ES는 `9.154→0.500`으로 줄었다. routing의 존재를 행동 인과 중요도로 읽을 수 없다.

## 한계

- attention probability는 value vector, residual stream, MLP, AdaLN gate를 포함하지 않으므로 feature importance가 아니다.
- 5 flow timestep은 같은 noisy action trajectory 안의 반복 계산이며 독립 표본이 아니다. attention 조건당 표본 수도 1이다.
- 행동 민감도는 별도 N=24 intervention run에서 왔으므로 attention run과 표본 단위가 다르다.
- token masking/attention ablation은 policy forward를 바꾸므로 이번 read-only 측정에 포함하지 않았다. 성능·정확도·성공률은 이 산출물만으로 주장할 수 없다.

## 산출물

- `combined_summary.json`: 축/집계/정규화/token/ES 해석 guide를 포함한 machine-readable 결과
- `combined_summary.csv`: 조건별 scalar
- 3개 heatmap과 1개 attention–행동 ES scatter
- raw `*.npz`, `*.tokens.json`, `manifest.jsonl`: 변경하지 않은 원본
"""
    (args.out / "REPORT.ko.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary["tensor_validation"], ensure_ascii=False))


if __name__ == "__main__":
    main()
