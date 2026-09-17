#!/usr/bin/env python3
"""Explicit one-way migration of positional LoRA checkpoints to schema v2."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from checkpoint_schema import build_checkpoint_metadata  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--targets-json", required=True,
                    help="JSON list of target module names in the legacy wrapper order")
    ap.add_argument("--adapter-skills", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--alpha", type=int, required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--sampler", required=True)
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--source-manifest-sha256", required=True)
    ap.add_argument("--update-index", type=int, required=True)
    args = ap.parse_args()

    blob = torch.load(args.input, map_location="cpu")
    old = blob.get("lora", {})
    if not old or not all(str(k).startswith("w") for k in old):
        raise SystemExit("input is not a positional legacy LoRA checkpoint")
    targets = json.loads(Path(args.targets_json).read_text())
    skills = [s for s in args.adapter_skills.split(",") if s]
    named = {}
    for index, target in enumerate(targets):
        for skill in skills:
            for kind in ("lora_A", "lora_B"):
                old_key = f"w{index}.{kind}.{skill}"
                if old_key not in old:
                    raise SystemExit(f"missing legacy tensor: {old_key}")
                named[f"{target}.{kind}.{skill}"] = old[old_key]
    if len(named) != len(old):
        raise SystemExit("legacy checkpoint contains unexpected positional tensors")
    blob["lora"] = named
    blob["metadata"] = build_checkpoint_metadata(
        adapter_skills=skills, targets=targets, rank=args.rank, alpha=args.alpha,
        base_model=args.base_model, sampler=args.sampler,
        config=blob.get("config", {}), source_commit=args.source_commit,
        source_manifest_sha256=args.source_manifest_sha256,
        update_index=args.update_index,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
