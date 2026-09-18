#!/usr/bin/env python3
"""Verify representative MP4 encoding with imageio-ffmpeg's bundled binary."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "results" / "videos")
root = parser.parse_args().root
rows = []
for path in sorted(root.glob("*.mp4")):
    proc = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path), "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    text = proc.stderr
    codec = re.search(r"Video:\s*([^,]+),\s*([^,(]+)", text)
    if codec is None:
        raise RuntimeError(f"could not identify video codec for {path}")
    row = {"path": str(path.relative_to(root.parent)),
           "codec": codec.group(1).strip(), "pixel_format": codec.group(2).strip(),
           "decode_returncode": proc.returncode}
    if row["codec"] not in ("h264", "h264 (High)") and not row["codec"].startswith("h264"):
        raise RuntimeError(f"not H.264: {row}")
    if row["pixel_format"] != "yuv420p":
        raise RuntimeError(f"not yuv420p: {row}")
    if proc.returncode != 0:
        raise RuntimeError(f"decode failed: {row}")
    rows.append(row)
(root / "encoding.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
