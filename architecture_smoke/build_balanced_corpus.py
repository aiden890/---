#!/usr/bin/env python3
"""Select a deterministic balanced view of a seed-major rollout corpus.

The collector stores ``seedN/<skill>.*`` once.  The VLM cache reader expects
``<skill>/seedN/<skill>.*``.  This script creates symlinks rather than copying
videos, and records exactly which offline sim-GT labels were selected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _has_proprio14(seed_dir, skill):
    path = seed_dir / f"{skill}_steps.jsonl"
    try:
        for line in path.open():
            row = json.loads(line)
            if isinstance(row.get("proprio"), list) and len(row["proprio"]) == 14:
                return True
    except (OSError, json.JSONDecodeError):
        pass
    return False


def build_balanced_corpus(source, dest, *, skills, positives=15, negatives=15):
    source, dest = Path(source).resolve(), Path(dest)
    rows = []
    for result_path in sorted(source.glob("seed*/results.json"),
                              key=lambda p: int(p.parent.name[4:])):
        seed = int(result_path.parent.name[4:])
        result = json.loads(result_path.read_text())
        rows.append((seed, result_path.parent, result))

    chosen = {}
    for skill in skills:
        eligible = [(s, d, r) for s, d, r in rows if _has_proprio14(d, skill)]
        pos = [(s, d) for s, d, r in eligible if r.get(skill, {}).get("success") is True]
        neg = [(s, d) for s, d, r in eligible if r.get(skill, {}).get("success") is False]
        if len(pos) < positives:
            raise ValueError(f"{skill}: positive {len(pos)} < target {positives}")
        if len(neg) < negatives:
            raise ValueError(f"{skill}: negative {len(neg)} < target {negatives}")
        chosen[skill] = (pos[:positives], neg[:negatives])

    dest.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(source),
        "targets": {"positive": positives, "negative": negatives},
        "label_source": "results.json success (sim predicate GT; offline only)",
        "eligibility": "every selected skill episode has per-step proprio[14]",
        "skills": {},
    }
    for skill, (pos, neg) in chosen.items():
        skill_dir = dest / skill
        skill_dir.mkdir(exist_ok=True)
        for old in skill_dir.glob("seed*"):
            if old.is_symlink():
                old.unlink()
        selected = pos + neg
        for seed, seed_dir in selected:
            (skill_dir / f"seed{seed}").symlink_to(seed_dir, target_is_directory=True)
        manifest["skills"][skill] = {
            "n_positive": len(pos), "n_negative": len(neg),
            "positive_seeds": [s for s, _ in pos],
            "negative_seeds": [s for s, _ in neg],
        }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--skills", default="grasp,move_holding,place")
    ap.add_argument("--positives", type=int, default=15)
    ap.add_argument("--negatives", type=int, default=15)
    args = ap.parse_args()
    manifest = build_balanced_corpus(
        args.source, args.dest,
        skills=tuple(x.strip() for x in args.skills.split(",") if x.strip()),
        positives=args.positives, negatives=args.negatives)
    for skill, d in manifest["skills"].items():
        print(f"{skill}: pos={d['n_positive']} neg={d['n_negative']}")
    print(f"wrote {Path(args.dest) / 'manifest.json'}")


if __name__ == "__main__":
    main()
