#!/usr/bin/env python3
"""Render the RL-environment reward timeline directly onto each source video."""
from __future__ import annotations

import argparse
import bisect
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FFMPEG = "/home/aiden/miniconda3/bin/ffmpeg"


def timeline_row_for_frame(timeline: list[dict], frame_index: int) -> dict:
    positions = [int(row["f"]) for row in timeline]
    index = max(0, bisect.bisect_right(positions, frame_index) - 1)
    return timeline[index]


def reward_lines(row: dict) -> list[str]:
    delta = row.get("d", {})
    term = float(delta.get("term", 0.0) or 0.0)
    hold = float(delta.get("hold", 0.0) or 0.0)
    penalty = float(delta.get("pen", 0.0) or 0.0)
    shaping = float(delta.get("shape", 0.0) or 0.0)
    step_delta = term + hold + penalty + shaping
    return [
        f"step {int(row['s'])}  |  delta {step_delta:+.3f}  |  cumulative {float(row.get('t', 0.0)):+.3f}",
        f"terminal {term:+.3f}   hold {hold:+.3f}   penalty {penalty:+.3f}   shaping {shaping:+.3f}",
    ]


def _font(size: int, bold: bool = False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def _truth(value) -> str:
    if value is True:
        return "T"
    if value is False:
        return "F"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def draw_overlay(frame, clip: dict, row: dict, labels: dict):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    width, height = image.size
    top_h, bottom_h = 62, 78
    draw.rectangle((0, 0, width, top_h), fill=(17, 20, 26, 224))
    draw.rectangle((0, height - bottom_h, width, height), fill=(17, 20, 26, 232))

    event = any(abs(float(row.get("d", {}).get(k, 0.0) or 0.0)) > 1e-12 for k in ("term", "hold", "pen", "shape"))
    success = float(row.get("d", {}).get("term", 0.0) or 0.0) > 0
    border = (54, 211, 153, 255) if success else ((248, 113, 113, 255) if event else (96, 165, 250, 220))
    draw.rectangle((1, 1, width - 2, height - 2), outline=border, width=4)

    title = f"{clip['label'].upper()}  |  {clip.get('case', '')}"
    draw.text((10, 7), title, font=_font(16, True), fill=(240, 244, 250, 255))
    phase = "성공 후 HOLD" if row.get("phase") == "post_success" else "스킬 수행 중"
    draw.text((10, 34), f"판정 목표: {clip.get('outcome')}  |  현재 구간: {phase}", font=_font(14), fill=(183, 203, 232, 255))

    lines = reward_lines(row)
    reward_color = (54, 211, 153, 255) if success else ((248, 113, 113, 255) if event else (235, 239, 245, 255))
    draw.text((10, height - 72), lines[0], font=_font(16, True), fill=reward_color)
    draw.text((10, height - 46), lines[1], font=_font(13), fill=(219, 225, 234, 255))

    predicates = row.get("p", {})
    keys = ["lid_grasped", "in_preplace_region", "lid_on_blender", "lid_upright_7deg", "gripper_lid_far_0.15"]
    parts = [f"{labels.get(k, k)}={_truth(predicates.get(k))}" for k in keys]
    draw.text((10, height - 23), "  |  ".join(parts), font=_font(11), fill=(180, 190, 204, 255))

    if row.get("inactive"):
        badge = "스킬 종료 후: 리워드 계산 중지"
        bbox = draw.textbbox((0, 0), badge, font=_font(14, True))
        x = width - (bbox[2] - bbox[0]) - 12
        draw.rounded_rectangle((x - 7, 33, width - 6, 58), radius=5, fill=(98, 74, 26, 235))
        draw.text((x, 35), badge, font=_font(14, True), fill=(255, 225, 150, 255))

    if row.get("end"):
        badge = f"종료: {row['end']['note']} / total {float(row['end']['total']):+.3f}"
        bbox = draw.textbbox((0, 0), badge, font=_font(14, True))
        x = max(8, width - (bbox[2] - bbox[0]) - 12)
        draw.rounded_rectangle((x - 7, 5, width - 6, 31), radius=5, fill=border)
        draw.text((x, 7), badge, font=_font(14, True), fill=(15, 18, 22, 255))

    composed = Image.alpha_composite(image, layer).convert("RGB")
    return cv2.cvtColor(np.asarray(composed), cv2.COLOR_RGB2BGR)


def render_clip(root: Path, clip: dict, labels: dict) -> dict:
    source = root / clip["mp4"]
    output_rel = str(Path(clip["mp4"]).with_name(Path(clip["mp4"]).stem + "_reward_overlay.mp4"))
    output = root / output_rel
    temporary = output.with_suffix(".mp4v.mp4")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or float(clip.get("fps", 20))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot write {temporary}")

    frames = 0
    event_frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        row = timeline_row_for_frame(clip["timeline"], frames)
        if any(abs(float(row.get("d", {}).get(k, 0.0) or 0.0)) > 1e-12 for k in ("term", "hold", "pen", "shape")):
            event_frames += 1
        writer.write(draw_overlay(frame, clip, row, labels))
        frames += 1
    capture.release()
    writer.release()

    subprocess.run([
        FFMPEG, "-y", "-v", "error", "-i", str(temporary), "-c:v", "libx264",
        "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an", str(output),
    ], check=True)
    temporary.unlink()
    clip["overlay_mp4"] = output_rel
    return {"source": str(source), "output": str(output), "frames": frames, "event_frames": event_frames}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "rlenv.json"))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    root = config_path.parent
    data = json.loads(config_path.read_text())
    results = [render_clip(root, clip, data["predicate_labels"]) for clip in data["clips"]]
    config_path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
