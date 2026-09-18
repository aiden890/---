#!/usr/bin/env python3
"""Compare measured serial and RLinf-parallel collector runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_run(root: Path) -> dict:
    root = Path(root)
    episodes = _rows(root / "episodes.jsonl")
    failures = _rows(root / "failures.jsonl")
    elapsed = 0.0
    if episodes:
        elapsed = max(float(row["finished_at"]) for row in episodes) - min(
            float(row["started_at"]) for row in episodes)
    peaks = [float(row.get("trainer_memory_after", {}).get("peak_allocated_gb", 0.0))
             for row in episodes]
    return {
        "run": str(root),
        "episodes": len(episodes),
        "elapsed_seconds": round(elapsed, 3),
        "episodes_per_hour": (round(len(episodes) * 3600.0 / elapsed, 3) if elapsed > 0 else None),
        "mean_episode_wall_seconds": (
            round(sum(float(row.get("timing", {}).get("wall_seconds", 0.0)) for row in episodes) /
                  len(episodes), 3) if episodes else None),
        "trainer_peak_allocated_gb": max(peaks, default=0.0),
        "failure_attempts": len(failures),
        "done": (root / "DONE").exists(),
        "failed": (root / "FAILED").exists(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("serial", type=Path)
    ap.add_argument("parallel", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    serial = summarize_run(args.serial)
    parallel = summarize_run(args.parallel)
    speedup = None
    if serial["elapsed_seconds"] and parallel["elapsed_seconds"]:
        speedup = serial["elapsed_seconds"] / parallel["elapsed_seconds"]
    report = {"serial": serial, "parallel": parallel,
              "parallel_speedup": (round(speedup, 3) if speedup is not None else None)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
