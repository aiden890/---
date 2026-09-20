#!/usr/bin/env python3
"""Build a zero-copy, filtered rollout staging tree for compressed transfer."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def prepare(run_dir: Path, stage_dir: Path, *, allowed_root: Path = Path("/results")) -> dict:
    run_dir = Path(run_dir).resolve()
    stage_dir = Path(stage_dir).resolve()
    results = Path(allowed_root).resolve()
    run_dir.relative_to(results)
    stage_dir.relative_to(results)
    if run_dir == stage_dir or run_dir in stage_dir.parents:
        raise ValueError("transfer staging directory must not contain the source run")
    rows = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text().splitlines()
            if line.strip()]
    groups = {}
    for row in rows:
        payload = row["trainer_payload"]
        groups.setdefault(str(payload["group_id"]), []).append(row)
    kept, dropped = [], []
    for group_id, members in groups.items():
        successes = [member.get("success") for member in members]
        if all(value is not None for value in successes) and not any(successes):
            dropped.append(group_id)
        else:
            kept.extend(members)
    if not kept:
        raise RuntimeError("rollout wave has no transferable groups")
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    linked_bytes = 0
    for row in kept:
        source = Path(row["trainer_payload"]["path"]).resolve()
        relative = source.relative_to(run_dir)
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = stage_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(str(source), str(destination))
        linked_bytes += source.stat().st_size
    with (stage_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for row in kept:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    result = {
        "run_dir": str(run_dir), "stage_dir": str(stage_dir),
        "groups_total": len(groups), "groups_transferred": len(groups) - len(dropped),
        "groups_dropped": dropped, "episodes_transferred": len(kept),
        "uncompressed_payload_bytes": linked_bytes,
    }
    (stage_dir / "transfer_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("stage_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.run_dir, args.stage_dir), sort_keys=True))


if __name__ == "__main__":
    main()
