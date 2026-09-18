#!/usr/bin/env python3
"""Aggregate and hard-gate the fixed-20 closed-loop measurement."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

SKILLS = ("GRASP_OBJECT", "MOVE_OBJECT", "PLACE_OBJECT")


def percentile(values, q):
    values = sorted(float(v) for v in values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def exact_mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    episodes = json.loads((root / "episodes.json").read_text())
    baseline_payload = json.loads(args.baseline.read_text())
    baseline = {int(e["seed"]): bool(e["official_success"])
                for e in baseline_payload["episodes"]}
    if [e["seed"] for e in episodes] != list(range(5000, 5020)):
        raise RuntimeError("fixed seed denominator is not exactly 5000..5019")

    async_events = Counter()
    rpc_status = Counter()
    control_errors = []
    for episode in episodes:
        trace_path = root / f"seed{episode['seed']}" / "trace.jsonl"
        for line in trace_path.read_text().splitlines():
            row = json.loads(line)
            if row.get("type") != "async_control":
                continue
            event = row.get("event")
            async_events[event] += 1
            if event == "rpc_done":
                rpc_status[str(row.get("status"))] += 1
                if row.get("status") != "ok":
                    control_errors.append({"seed": episode["seed"], **row})
            if event in ("rpc_error", "deadline_miss", "queue_violation"):
                control_errors.append({"seed": episode["seed"], **row})

    planner_valid = sum(bool(e["planner"]["valid"]) for e in episodes)
    official = sum(bool(e["task_success"]) for e in episodes)
    reaches = {skill: sum(any(s["skill"] == skill for s in e["skills"]) for e in episodes)
               for skill in SKILLS}
    skill_success = {
        skill: sum(any(s["skill"] == skill and s["status"] == "SUCCESS"
                       for s in e["skills"]) for e in episodes)
        for skill in SKILLS
    }
    all_policy_latencies = [v for e in episodes for v in e["policy"]["latencies_ms"]]
    ours = {int(e["seed"]): bool(e["task_success"]) for e in episodes}
    baseline_only = sum(baseline[s] and not ours[s] for s in ours)
    planned_only = sum(ours[s] and not baseline[s] for s in ours)
    paired_same = sum(ours[s] == baseline[s] for s in ours)

    mechanics = {
        "completed_scoring": len(episodes),
        "planner_calls_exactly_one": all(e["planner"]["calls"] == 1 for e in episodes),
        "runtime_boundary_planner_calls": sum(e["mechanics"]["runtime_boundary_planner_calls"] for e in episodes),
        "fallback_calls": sum(e["mechanics"]["fallback_calls"] for e in episodes),
        "repair_calls": sum(e["mechanics"]["repair_calls"] for e in episodes),
        "sequence_corrections": sum(e["mechanics"]["sequence_corrections"] for e in episodes),
        "action_before_plan": sum(e["mechanics"]["action_before_plan"] for e in episodes),
        "nan_violations": sum(e["mechanics"]["nan_violations"] for e in episodes),
        "policy_deadline_misses": sum(e["policy"]["deadline_misses"] for e in episodes),
        "rpc_or_control_errors": len(control_errors),
        "queue_violations": async_events.get("queue_violation", 0),
        "latest_only_superseded_drops": async_events.get("drop", 0),
    }
    gate = (
        len(episodes) == 20 and planner_valid == 20 and
        mechanics["planner_calls_exactly_one"] and
        mechanics["runtime_boundary_planner_calls"] == 0 and
        mechanics["fallback_calls"] == 0 and mechanics["repair_calls"] == 0 and
        mechanics["sequence_corrections"] == 0 and mechanics["action_before_plan"] == 0 and
        mechanics["nan_violations"] == 0 and mechanics["policy_deadline_misses"] == 0 and
        mechanics["rpc_or_control_errors"] == 0 and mechanics["queue_violations"] == 0
    )
    aggregate = {
        "verdict": "TASK_ACCURACY_MEASURED" if gate else "NOT_READY",
        "production_ready_claimed": False,
        "source_commit": "45280448bd9f650c3cfd94f26939a8830374a05b",
        "seeds": list(range(5000, 5020)),
        "rates": {
            "planner_valid": {"count": planner_valid, "denominator": 20, "rate": planner_valid / 20},
            "boundary_reach": {skill: {"count": reaches[skill], "denominator": 20,
                                        "rate": reaches[skill] / 20} for skill in SKILLS},
            "skill_success": {skill: {"count": skill_success[skill], "denominator": 20,
                                       "rate": skill_success[skill] / 20} for skill in SKILLS},
            "official_full_task": {"count": official, "denominator": 20, "rate": official / 20},
        },
        "baseline_comparison": {
            "baseline": {"count": sum(baseline.values()), "denominator": 20, "rate": sum(baseline.values()) / 20},
            "planned_runtime": {"count": official, "denominator": 20, "rate": official / 20},
            "delta_successes": official - sum(baseline.values()),
            "delta_percentage_points": (official - sum(baseline.values())) * 5.0,
            "paired_same": paired_same, "baseline_only": baseline_only,
            "planned_only": planned_only,
            "mcnemar_exact_two_sided_p": exact_mcnemar_p(baseline_only, planned_only),
            "interpretation": "No aggregate improvement; n=20 does not support an improvement claim.",
        },
        "latency_ms": {
            "planner_min": min(e["planner"]["latency_ms"] for e in episodes),
            "planner_mean": sum(e["planner"]["latency_ms"] for e in episodes) / 20,
            "planner_max": max(e["planner"]["latency_ms"] for e in episodes),
            "policy_count": len(all_policy_latencies),
            "policy_p50": percentile(all_policy_latencies, 0.5),
            "policy_p95": percentile(all_policy_latencies, 0.95),
            "policy_max": max(all_policy_latencies),
        },
        "mechanics": mechanics,
        "async_event_counts": dict(sorted(async_events.items())),
        "rpc_status_counts": dict(sorted(rpc_status.items())),
        "failure_stages": dict(sorted(Counter(e.get("failure_stage") or "none" for e in episodes).items())),
        "official_success_seeds": [e["seed"] for e in episodes if e["task_success"]],
    }
    (root / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")

    with (root / "planner_raw.jsonl").open("w") as handle:
        for e in episodes:
            handle.write(json.dumps({"seed": e["seed"], "raw_json": e["planner"]["raw_json"]}) + "\n")

    videos = root / "videos"
    videos.mkdir(exist_ok=True)
    success_eps = [e for e in episodes if e["task_success"]]
    failure_eps = [e for e in episodes if not e["task_success"]]
    chosen = []
    if success_eps:
        chosen.append((success_eps[0], "success"))
    grasp_failure = next((e for e in failure_eps if e.get("failure_stage") == "GRASP_OBJECT"), None)
    late_failure = next((e for e in failure_eps if e.get("failure_stage") in ("MOVE_OBJECT", "PLACE_OBJECT", "official_predicate")), None)
    for episode, label in ((grasp_failure, "execution_failure_grasp"),
                           (late_failure, "execution_failure_late")):
        if episode is not None and all(existing[0]["seed"] != episode["seed"] for existing in chosen):
            chosen.append((episode, label))
    for episode in failure_eps:
        if len(chosen) >= 3:
            break
        if all(existing[0]["seed"] != episode["seed"] for existing in chosen):
            chosen.append((episode, "execution_failure"))
    manifest = []
    for episode, label in chosen[:3]:
        src = root / f"seed{episode['seed']}" / "episode.mp4"
        dst = videos / f"{label}_seed{episode['seed']}.mp4"
        shutil.copyfile(src, dst)
        manifest.append({"seed": episode["seed"], "label": label,
                         "path": str(dst.relative_to(root)), "sha256": sha256(dst)})
    (videos / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    lines = [
        "# Generic planner fixed-20 closed-loop task accuracy", "",
        f"Verdict: `{aggregate['verdict']}`", "",
        "This is a task-accuracy measurement, not a `PRODUCTION_READY` claim.", "",
        "## Result", "",
        f"- Planner-valid: {planner_valid}/20 ({planner_valid/20:.0%}).",
        f"- Boundary reach: GRASP {reaches['GRASP_OBJECT']}/20, MOVE {reaches['MOVE_OBJECT']}/20, PLACE {reaches['PLACE_OBJECT']}/20.",
        f"- Obs-only boundary success: GRASP {skill_success['GRASP_OBJECT']}/20, MOVE {skill_success['MOVE_OBJECT']}/20, PLACE {skill_success['PLACE_OBJECT']}/20.",
        f"- Official RoboCasa full-task success: {official}/20 ({official/20:.0%}); seeds {aggregate['official_success_seeds']}.",
        f"- Existing paired-seed baseline: {sum(baseline.values())}/20; delta {official-sum(baseline.values()):+d}/20 ({aggregate['baseline_comparison']['delta_percentage_points']:+.1f} pp).",
        f"- Paired discordance: baseline-only {baseline_only}, planned-runtime-only {planned_only}; exact McNemar p={aggregate['baseline_comparison']['mcnemar_exact_two_sided_p']:.4g}.",
        "- Interpretation: no aggregate improvement; the 20-seed result does not support an improvement claim.", "",
        "## Mechanics hard gate", "",
        f"- Exactly one planner call per episode: {mechanics['planner_calls_exactly_one']}; runtime boundary calls {mechanics['runtime_boundary_planner_calls']}.",
        f"- Fallback/repair/sequence correction/action-before-plan: {mechanics['fallback_calls']}/{mechanics['repair_calls']}/{mechanics['sequence_corrections']}/{mechanics['action_before_plan']}.",
        f"- NaN/policy deadline/RPC-control/queue violations: {mechanics['nan_violations']}/{mechanics['policy_deadline_misses']}/{mechanics['rpc_or_control_errors']}/{mechanics['queue_violations']}.",
        f"- Latest-only superseded boundary observations: {mechanics['latest_only_superseded_drops']} (expected coalescing, not queue violations).",
        f"- Completed scoring: {mechanics['completed_scoring']}/20.", "",
        "## Latency", "",
        f"- Planner latency min/mean/max: {aggregate['latency_ms']['planner_min']:.1f}/{aggregate['latency_ms']['planner_mean']:.1f}/{aggregate['latency_ms']['planner_max']:.1f} ms.",
        f"- Policy forward p50/p95/max over {aggregate['latency_ms']['policy_count']} chunks: {aggregate['latency_ms']['policy_p50']:.1f}/{aggregate['latency_ms']['policy_p95']:.1f}/{aggregate['latency_ms']['policy_max']:.1f} ms (800 ms contract).", "",
        "## Protocol and evidence", "",
        "- Exact source commit: `45280448bd9f650c3cfd94f26939a8830374a05b`.",
        "- Fixed target-split seeds: `5000..5019`; all failures remain in denominator 20.",
        "- Generic `Qwen/Qwen3-VL-4B-Instruct@ebb281ec...` planner runs once from the reset 3-camera+proprio snapshot; direct raw JSON is in `planner_raw.jsonl` and each `seed*/result.json`.",
        "- Xiaomi base policy and obs-only verifier receive no simulator predicates. Simulator predicates are read only after plan consumption for offline per-seed scoring.",
        "- `per_seed.csv`, `episodes.json`, `seed*/trace.jsonl`, and `seed*/result.json` preserve reset hashes, canonical plans, transitions, actions, latencies, failure stages, and official outcomes.",
        "- Three representative videos are listed in `videos/manifest.json`; no planning-failure video exists because all 20 plans were valid.",
    ]
    (root.parent / "REPORT.md").write_text("\n".join(lines) + "\n")

    artifact_paths = [p for p in root.rglob("*")
                      if p.is_file() and p.name not in ("artifacts.sha256", "finalize.log")]
    artifact_paths.extend(
        p for p in root.parent.iterdir()
        if p.is_file() and p.name not in ("artifacts.sha256",) and p.suffix != ".pyc")
    with (root / "artifacts.sha256").open("w") as handle:
        for path in sorted(artifact_paths):
            handle.write(f"{sha256(path)}  {path.relative_to(root.parent)}\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
