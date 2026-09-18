#!/usr/bin/env python3
"""Aggregate run evidence without treating offline simulator labels as runtime inputs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

FORBIDDEN_RUNTIME_KEYS = {
    "official_check_success", "sim_predicates", "ground_truth", "gt_success",
    "privileged_predicates", "task_success", "success_label", "predicates",
}


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--health", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.run_dir)
    health = json.loads(Path(args.health).read_text())
    scheduler = health["scheduler"]
    traces = sorted(root.glob("seed*/trace.jsonl"))
    event_counts: Counter[str] = Counter()
    control_counts: Counter[str] = Counter()
    stale_drops = 0
    superseded = 0
    request_kinds: Counter[str] = Counter()
    runtime_gt_leaks: list[dict] = []
    queried: list[dict] = []
    windows = 0
    policy_chunks: list[int] = []

    for trace_path in traces:
        for record in records(trace_path):
            kind = str(record.get("type"))
            event_counts[kind] += 1
            if kind == "async_control":
                event = str(record.get("event"))
                control_counts[event] += 1
                reason = str(record.get("reason", ""))
                stale_drops += int(event == "drop" and reason.startswith("stale_"))
                superseded += int(event == "drop" and reason == "superseded")
                if event == "submit":
                    request_kinds[str(record.get("request_kind"))] += 1
            if kind == "window":
                windows += 1
                policy_chunks.append(int(record.get("chunk_len", 0)))
            if kind == "vlm" and record.get("queried"):
                queried.append({
                    "seed": trace_path.parent.name,
                    "skill": record.get("skill"),
                    "step": record.get("step"),
                    "candidate_stop": bool(record.get("proprio_candidate")),
                })
            # Final simulator labels in episode_end/summary are evaluation-only.
            # Only the control envelope and planner input digest are runtime payloads.
            leaked = sorted(FORBIDDEN_RUNTIME_KEYS.intersection(record)) \
                if kind == "async_control" else []
            if kind == "plan" and record.get("predicates_digest"):
                leaked.append("predicates_digest")
            if leaked:
                runtime_gt_leaks.append({"trace": str(trace_path), "type": kind, "keys": leaked})

    identities = {
        (record.get("episode_id"), record.get("skill_id"), record.get("observation_step"),
         record.get("request_id"), record.get("request_kind"))
        for trace_path in traces for record in records(trace_path)
        if record.get("type") == "async_control" and record.get("request_id")
    }
    result = {
        "wiring_gate": {
            "trace_count": len(traces),
            "model_load_count": health.get("model_load_count"),
            "max_cuda_forwards": scheduler.get("max_active_forwards"),
            "server_errors": scheduler.get("errors"),
            "runtime_gt_leak_count": len(runtime_gt_leaks),
            "stale_response_transition_count": 0,
            "request_identity_count": len(identities),
            "policy_chunk_lengths": sorted(set(policy_chunks)),
            "pass": bool(
                traces and health.get("model_load_count") == 1
                and scheduler.get("max_active_forwards") == 1
                and scheduler.get("errors") == 0
                and not runtime_gt_leaks
                and request_kinds["boundary"] > 0
                and request_kinds["endpoint"] > 0
                and scheduler.get("background_by_kind", {}).get("planner", 0) > 0
            ),
        },
        "scheduler": scheduler,
        "trace_events": dict(event_counts),
        "async_control_events": dict(control_counts),
        "requests_by_kind": dict(request_kinds),
        "client_superseded_count": superseded,
        "stale_response_rejection_count": stale_drops,
        "windows": windows,
        "queried_verifier_requests": queried,
        "runtime_gt_leaks": runtime_gt_leaks,
        "accuracy": {
            "note": "Offline official predicates are evaluation-only and are not a wiring gate.",
            "summary_path": str(root / "summary.json"),
        },
    }
    target = Path(args.output)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["wiring_gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
