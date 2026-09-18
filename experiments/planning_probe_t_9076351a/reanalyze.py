#!/usr/bin/env python3
"""Recompute metrics from preserved raw generations without model inference."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


import probe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    args = ap.parse_args()
    out = Path(args.results)
    records = json.loads((out / "cases.json").read_text(encoding="utf-8"))
    full_keys = set(probe.score_plan(None))
    next_keys = {f"next_{key}" for key in probe.validate_call(None)} | {"next_first_skill_correct"}
    for row in records:
        parsed, _ = probe.direct_json(row["raw_text"])
        if row["mode"] == "explicit_full_plan":
            for key in full_keys:
                row.pop(key, None)
            row.update(probe.score_plan(parsed))
        else:
            for key in next_keys:
                row.pop(key, None)
            validity = probe.validate_call(parsed)
            row.update({f"next_{key}": value for key, value in validity.items()})
            row["next_first_skill_correct"] = (
                isinstance(parsed, dict) and parsed.get("name") == probe.EXPECTED[0]
            )
    (out / "cases.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    flat = [{k: (json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v)
             for k, v in row.items() if k not in {"prompt", "raw_text"}} for row in records]
    with (out / "cases.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=sorted({k for row in flat for k in row}), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)
    images = json.loads((out / "image_manifest.json").read_text(encoding="utf-8"))
    probe.write_summary(out, records, images, argparse.Namespace(max_new_tokens=512))


if __name__ == "__main__":
    main()
