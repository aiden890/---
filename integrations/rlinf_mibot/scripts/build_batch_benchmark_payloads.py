#!/usr/bin/env python3
"""Clone one mixed rollout group into isolated 1/2/4-group compute benchmarks."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch


def parse_sizes(raw: str) -> list[int]:
    tokens = raw.split(",")
    if any(not value.strip() for value in tokens):
        raise ValueError("benchmark sizes must be a comma-separated integer list")
    sizes = [int(value.strip()) for value in tokens]
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("benchmark sizes must be positive integers")
    if len(sizes) != len(set(sizes)):
        raise ValueError("benchmark sizes must not contain duplicates")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--sizes", default="1,2,4")
    parser.add_argument("--prefix", default="batchbench")
    args = parser.parse_args()
    source = args.source.resolve()
    output_root = args.output_root.resolve()
    sizes = parse_sizes(args.sizes)
    rows = [json.loads(line) for line in (source / "episodes.jsonl").read_text().splitlines()
            if line.strip()]
    group_ids = {str(row["group_id"]) for row in rows}
    if len(group_ids) != 1:
        raise ValueError(f"source must contain exactly one group, got {sorted(group_ids)}")
    rewards = [float(row["reward"]) for row in rows]
    if any(not math.isfinite(reward) for reward in rewards):
        raise ValueError("source group contains a non-finite reward")
    success_values = [row.get("success") for row in rows]
    if any(value is not None and not isinstance(value, bool) for value in success_values):
        raise ValueError("source success values must be boolean or null")
    if all(value is not None for value in success_values):
        homogeneous = len({bool(value) for value in success_values}) < 2
    else:
        homogeneous = len(set(rewards)) < 2
    if homogeneous:
        raise ValueError("source group is homogeneous and would be discarded by GRPO")

    reports = []
    for size in sizes:
        destination = (output_root / f"{args.prefix}-g{size}").resolve()
        if not destination.is_relative_to(output_root):
            raise ValueError(f"benchmark destination escapes output root: {destination}")
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite benchmark directory: {destination}")
        output_rows = []
        for replica in range(size):
            for original in rows:
                row = copy.deepcopy(original)
                old_path = Path(original["trainer_payload"]["path"]).resolve()
                if not old_path.is_relative_to(source):
                    raise ValueError(f"source payload is outside source directory: {old_path}")
                actual_hash = hashlib.sha256(old_path.read_bytes()).hexdigest()
                if actual_hash != original["trainer_payload"].get("sha256"):
                    raise ValueError(f"source payload hash mismatch: {old_path}")
                payload = torch.load(old_path, weights_only=True)
                if len(payload.get("trajectories", {})) != 1:
                    raise ValueError(f"expected one trajectory per payload: {old_path}")
                old_id, chunks = next(iter(payload["trajectories"].items()))
                if [str(old_id)] != list(map(str, original["trainer_payload"]["trajectory_ids"])):
                    raise ValueError(f"trajectory metadata mismatch: {old_path}")
                if str(payload.get("group_id")) != str(original["group_id"]):
                    raise ValueError(f"group metadata mismatch: {old_path}")
                if len(chunks) != int(original["trainer_payload"]["n_chunks"]):
                    raise ValueError(f"chunk metadata mismatch: {old_path}")
                new_id = f"{old_id}__batchrep{replica}"
                new_group = f"{original['group_id']}__batchrep{replica}"
                payload["trajectories"] = {new_id: chunks}
                payload["group_id"] = new_group
                new_path = (destination / row["config_id"] / "payloads" /
                            f"{old_path.stem}__batchrep{replica}.pt").resolve()
                if not new_path.is_relative_to(destination):
                    raise ValueError(f"payload destination escapes benchmark directory: {new_path}")
                new_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, new_path)
                digest = hashlib.sha256(new_path.read_bytes()).hexdigest()
                row.update({"group_id": new_group,
                            "job_id": f"{row['job_id']}__batchrep{replica}"})
                row["trainer_payload"].update({
                    "path": str(new_path), "sha256": digest,
                    "trajectory_ids": [new_id], "group_id": new_group,
                })
                row["artifacts"]["trainer_store"] = str(new_path)
                output_rows.append(row)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "episodes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in output_rows))
        (destination / "audit.json").write_text(json.dumps({"pass": True}) + "\n")
        (destination / "DONE").write_text(json.dumps({"status": "done"}) + "\n")
        reports.append({"groups": size, "trajectories": len(output_rows),
                        "chunks": sum(int(row["trainer_payload"]["n_chunks"])
                                      for row in output_rows),
                        "path": str(destination)})
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
