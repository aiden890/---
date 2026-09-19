#!/usr/bin/env python3
"""Render cumulative GRASP reward sums onto rollout videos."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FFMPEG = "/home/aiden/miniconda3/bin/ffmpeg"
FFPROBE = "/home/aiden/miniconda3/bin/ffprobe"


def reward_transition_frame(success_step: int, video_stride: int) -> int:
    """Map an env success step to the forced/sampled video frame index.

    Rollouts save frame 0, every ``video_stride`` env steps, and the terminal
    success frame even when it falls between sampled steps.
    """
    if success_step < 1:
        raise ValueError("success_step must be positive")
    if video_stride < 1:
        raise ValueError("video_stride must be positive")
    return 1 + (success_step - 1) // video_stride


def build_record(
    row: dict,
    source_mp4: str,
    overlay_mp4: str,
    *,
    video_stride: int,
    video_fps: int,
) -> dict:
    success_step = row.get("hold_stats", {}).get("success_step")
    reward_frame = (
        reward_transition_frame(int(success_step), video_stride)
        if row.get("success") and success_step is not None
        else None
    )
    return {
        "job_id": row["job_id"],
        "seed": int(row["seed"]),
        "outcome": "SUCCESS" if row.get("success") else "TIMEOUT",
        "success": bool(row.get("success")),
        "steps": int(row["steps"]),
        "success_step": int(success_step) if success_step is not None else None,
        "reward_sum": float(row["reward"]),
        "reward_frame": reward_frame,
        "reward_time_seconds": reward_frame / video_fps if reward_frame is not None else None,
        "video_stride": int(video_stride),
        "video_fps": int(video_fps),
        "source_mp4": source_mp4,
        "mp4": overlay_mp4,
    }


def _drawtext(text: str, *, y: int, size: int, color: str, enable: str | None = None) -> str:
    escaped = text.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
    spec = (
        f"drawtext=fontfile={FONT_BOLD}:text='{escaped}':x=24:y={y}:"
        f"fontsize={size}:fontcolor={color}"
    )
    if enable:
        spec += f":enable='{enable}'"
    return spec


def build_ffmpeg_filter(record: dict) -> str:
    filters = [
        "drawbox=x=10:y=10:w=550:h=103:color=0x0c1018@0.88:t=fill",
        _drawtext(
            f"GRASP rollout · seed {record['seed']} · {record['outcome']}",
            y=19,
            size=16,
            color="0xebf0f8",
        ),
    ]
    reward_frame = record["reward_frame"]
    if reward_frame is None:
        filters.extend(
            [
                _drawtext("누적 리워드 합계 0.000000", y=46, size=26, color="0x94a3b8"),
                _drawtext("성공 조건 미충족 · 지급 없음", y=80, size=13, color="0xf8a8a8"),
            ]
        )
    else:
        before = f"lt(n\\,{reward_frame})"
        after = f"gte(n\\,{reward_frame})"
        filters.extend(
            [
                _drawtext("누적 리워드 합계 0.000000", y=46, size=26, color="0x94a3b8", enable=before),
                _drawtext(
                    f"성공 조건 대기 · 지급 예정 step {record['success_step']}",
                    y=80,
                    size=13,
                    color="0xcbd5e1",
                    enable=before,
                ),
                _drawtext(
                    f"누적 리워드 합계 {record['reward_sum']:.6f}",
                    y=46,
                    size=26,
                    color="0x34d399",
                    enable=after,
                ),
                _drawtext(
                    f"성공 terminal 지급 · step {record['success_step']} · +{record['reward_sum']:.6f}",
                    y=80,
                    size=13,
                    color="0x6ee7b7",
                    enable=after,
                ),
            ]
        )
    return ",".join(filters)


def _probe_video(path: Path) -> dict:
    result = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = (int(part) for part in stream["r_frame_rate"].split("/"))
    return {
        "frames": int(stream["nb_read_frames"]),
        "fps": numerator / denominator,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }


def render_video(source: Path, output: Path, record: dict) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            build_ffmpeg_filter(record),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(output),
        ],
        check=True,
    )
    return _probe_video(output)


def _source_for_row(source_root: Path, row: dict) -> Path:
    matches = list(source_root.rglob(f"{row['job_id']}/attempt-*/rollout.mp4"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one video for {row['job_id']}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=20)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.episodes.read_text().splitlines() if line.strip()]
    if len(rows) != 10:
        raise RuntimeError(f"expected exactly 10 rollouts, found {len(rows)}")

    records = []
    for row in rows:
        source = _source_for_row(args.source_root, row)
        output_name = f"grasp_seed_{int(row['seed'])}_reward_sum.mp4"
        output = args.output_dir / output_name
        record = build_record(
            row,
            str(row.get("artifacts", {}).get("video") or source),
            f"media/grasp_reward_rollouts/{output_name}",
            video_stride=args.video_stride,
            video_fps=args.video_fps,
        )
        record["video"] = render_video(source, output, record)
        records.append(record)

    source_parameters = rows[0]["parameters"]
    trainer_payload = rows[0]["trainer_payload"]
    payload = {
        "schema_version": 1,
        "title": "변경된 GRASP reward 환경 · rollout 10개",
        "source_run": args.source_root.name,
        "source_result_dir": f"spark2:/home/csi-agent-dgx_spark2/workspace/rlinf-mibot-prep/results/{args.source_root.name}",
        "policy_version": int(trainer_payload["policy_version"]),
        "policy_hash": trainer_payload["policy_hash"],
        "configuration": source_parameters,
        "reward_semantics": "GRASP 20-step stable success terminal × 0.998^step; milestones disabled",
        "rollouts": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"videos": len(records), "manifest": str(args.manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
