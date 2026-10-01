"""Extract released-cup successes, aligned to a held-mug action boundary."""

import argparse
import json
import hashlib
import os
from pathlib import Path
import numpy as np

INSTRUCTION = "Place the mug you are holding upright on the coffee machine tray directly under the dispenser, then release the mug."


def build(source, dest):
    manifests = []
    rejected = []
    for path in sorted(source.glob("*/bc/manifest.json")):
        m = json.loads(path.read_text())
        trace = json.loads((path.parent.parent / "trace.json").read_text())
        end = m["metrics"]["cup_placement_step"]
        if end is None:
            continue
        if any(r["coffee_machine_on"] for r in trace[:end]):
            rejected.append(
                dict(episode=path.parent.parent.name, reason="button_pressed_before_cup_placement")
            )
            continue
        starts = [r["step"] for r in m["samples"] if r["skill"] == "cup_placement"]
        eligible = [
            s
            for s in starts
            if s > 0 and trace[s - 1]["mug_grasped"] and trace[s - 1]["mug_motion"] > 0.1
        ]
        if not eligible:
            rejected.append(
                dict(episode=path.parent.parent.name, reason="no_observed_held_mug_chunk_boundary")
            )
            continue
        start = min(eligible)
        out = dest / path.parent.parent.name / "bc"
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        actions = np.load(path.parent.parent / "actions.npy", allow_pickle=False)
        for row in m["samples"]:
            s = row["step"]
            if row["skill"] != "cup_placement" or not start <= s < end:
                continue
            src = path.parent / row["observation"]
            target = out / row["observation"]
            if not target.exists():
                os.link(src, target)
            n = min(50, end - s)
            a = np.zeros((50, 12), np.float32)
            a[:n] = actions[s : s + n]
            np.savez_compressed(out / row["actions"], actions=a, valid=np.arange(50) < n)
            rows.append(row)
        data = dict(
            m,
            source_prompt=m["prompt"],
            prompt=INSTRUCTION,
            samples=rows,
            segments=[dict(skill="cup_placement", start=start, end=end)],
            dataset_task="PrepareCoffeeCupPlacement",
            precondition="native mug_grasped and displacement>0.10m at a captured 16-step boundary",
            label_source="successful_executed_policy_actions_relabelled_with_cup_only_instruction",
            source_episode=str(path.parent.parent),
            source_policy_version=json.loads((path.parent.parent / "result.json").read_text()).get(
                "policy_version", 0
            ),
            source_trace_sha256=hashlib.sha256(
                (path.parent.parent / "trace.json").read_bytes()
            ).hexdigest(),
        )
        (out / "manifest.json").write_text(json.dumps(data, indent=2))
        manifests.append(
            dict(
                episode=path.parent.parent.name,
                source_model=m["model"],
                seed=m["seed"],
                start=start,
                end=end,
                samples=len(rows),
            )
        )
    report = dict(
        instruction=INSTRUCTION,
        episodes=len(manifests),
        samples=sum(m["samples"] for m in manifests),
        accepted=manifests,
        rejected=rejected,
    )
    (dest / "dataset.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("destination")
    a = p.parse_args()
    dest = Path(a.destination)
    dest.mkdir(parents=True, exist_ok=True)
    print(json.dumps(build(Path(a.source), dest)))
