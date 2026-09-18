#!/usr/bin/env python3
"""Fixed-seed production-path CloseBlenderLid accuracy measurement.

Runs one strict generic-Qwen plan after reset, then consumes that immutable plan
through the Xiaomi policy and obs-only verifier. Simulator predicates are read
only after execution for offline scoring.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path

ARCH = Path("/pkg") if Path("/pkg").is_dir() else Path(__file__).resolve().parents[3] / "architecture_smoke"
if str(ARCH) not in sys.path:
    sys.path.insert(0, str(ARCH))

import bindings
from environment import RoboCasaEnvironment
from executor import ExecutionManager
from obs_verifier import ObsInput
from policy import BasePolicyClient
from pre_episode_planner import PlanRejected, PreEpisodePlanner, observation_fingerprint
from runtime_contract import CHUNK_DEADLINE_MS, POLICY_CHUNK_ACTIONS, as_dict as runtime_contract_dict
from schemas import AdapterMode, PlannerContext
from skills import SkillRegistry
from trace import Trace
from vlm_backends import RemoteVLMScorerBackend

SEEDS = tuple(range(5000, 5020))
SKILLS = ("GRASP_OBJECT", "MOVE_OBJECT", "PLACE_OBJECT")


class MeasuredPlannerService:
    def __init__(self, backend):
        self.backend = backend
        self.calls = 0
        self.latencies_ms: list[float] = []
        self.raw_outputs: list[object] = []

    def complete(self, request, timeout_s):
        self.calls += 1
        started = time.monotonic()
        response = self.backend.complete(request, timeout_s)
        self.latencies_ms.append((time.monotonic() - started) * 1000.0)
        self.raw_outputs.append(response.payload.get("text"))
        return response


class MeasuredPolicy:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.latencies_ms: list[float] = []
        self.nan_violations = 0
        self.action_before_plan = 0
        self.plan_validated = False

    def infer(self, pin):
        if not self.plan_validated:
            self.action_before_plan += 1
            raise RuntimeError("policy inference attempted before strict plan validation")
        started = time.monotonic()
        out = self.inner.infer(pin)
        self.latencies_ms.append((time.monotonic() - started) * 1000.0)
        self.calls += 1
        try:
            import numpy as np
            if not bool(np.isfinite(out.action_chunk).all()):
                self.nan_violations += 1
                raise RuntimeError("non-finite policy action")
        except ImportError:
            for row in out.action_chunk:
                if not all(math.isfinite(float(value)) for value in row):
                    self.nan_violations += 1
                    raise RuntimeError("non-finite policy action")
        return out

    def provenance(self):
        return self.inner.provenance()


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    pos = (len(ordered) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def failure_from(summary):
    if summary.get("task_success"):
        return None, None
    if summary.get("terminal") == "plan_rejected":
        return "planning", summary.get("plan_error")
    skills = summary.get("skills", [])
    for skill in skills:
        if skill.get("status") != "SUCCESS":
            return skill.get("skill", "execution"), skill.get("reason") or skill.get("terminated_by")
    if skills:
        return "official_predicate", "official full-task predicate false after canonical plan"
    return "runtime", summary.get("error", "episode did not execute")


def run_seed(args, registry, base_policy, backend, operating_points, seed):
    seed_dir = args.out / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    trace = Trace(seed_dir / "trace.jsonl", {
        "seed": seed, "task": "CloseBlenderLid",
        "source_commit": args.source_commit,
        "planner_kind": PreEpisodePlanner.kind,
        "verifier_kind": "obs_vlm", "runtime_gt_input": False,
    })
    env = None
    measured_policy = MeasuredPolicy(base_policy)
    measured_service = MeasuredPlannerService(backend)
    planner = PreEpisodePlanner(registry, measured_service, episode_id=f"seed{seed}", timeout_s=args.planner_timeout)
    started = time.monotonic()
    summary = {
        "seed": seed, "goal": bindings.GOAL, "terminal": "runtime_error",
        "planner_calls": 0, "runtime_boundary_planner_calls": 0,
        "steps_used": 0, "task_success": False, "skills": [],
    }
    try:
        env = RoboCasaEnvironment(
            seed=seed, split=args.split, replan_steps=POLICY_CHUNK_ACTIONS,
            obs_history=4, obs_interval=2, crop_ratio=0.95,
            video_stride=args.video_stride, video_fps=args.video_fps)
        env.reset()
        reset_obs = env.obs_for_verifier()
        reset_hash = observation_fingerprint(reset_obs)
        (seed_dir / "scene.json").write_text(json.dumps(env.scene_meta(), indent=2, default=str))
        context = PlannerContext(
            goal=bindings.GOAL, predicates={}, skill_catalog=registry.names,
            step_budget_remaining=args.episode_budget, planner_calls=0,
            observation=reset_obs)
        try:
            canonical = planner.plan_episode(context)
            planner.validate_execution_snapshot(canonical, env.obs_for_verifier())
            if sum(call.budget or 0 for call in canonical.calls) > args.episode_budget:
                raise PlanRejected("canonical full plan exceeds episode budget")
        except PlanRejected as exc:
            summary.update({
                "terminal": "plan_rejected", "planner_calls": planner.calls,
                "plan_error": str(exc), "reset_snapshot_sha256": reset_hash,
                "official_predicate": False, "final_predicates": {},
            })
            trace.episode_end(terminal="plan_rejected", planner_calls=planner.calls,
                              steps_used=0, task_success=False)
        else:
            measured_policy.plan_validated = True
            strict_plan = [call.as_dict() for call in canonical.calls]
            manager = ExecutionManager(
                registry, measured_policy, env, trace, AdapterMode.DISABLED,
                vlm_backend=backend, event_gated=not args.no_event_gate,
                synchronous_verifier=args.sync_verifier,
                episode_id=f"seed{seed}", verifier_operating_points=operating_points)
            skills = []
            summary["skills"] = skills
            steps_used = 0
            for boundary_index, call in enumerate(canonical.calls):
                trace.plan(boundary_index, env.observation_ref(), {}, call.as_dict(),
                           registry.render(call), "canonical pre-episode plan",
                           planner.kind)
                result = manager.execute(call)
                row = result.as_dict()
                skills.append(row)
                steps_used += result.steps
                summary["steps_used"] = steps_used
                trace.skill_result(row, next_skill=(
                    "advance_preplanned" if boundary_index + 1 < len(canonical.calls) else None))
            # Privileged state is read only here, after all runtime decisions.
            final_predicates = env.predicates()
            official = bool(final_predicates.get("official_check_success"))
            summary.update({
                "terminal": "plan_consumed", "planner_calls": planner.calls,
                "steps_used": steps_used, "task_success": official,
                "official_predicate": official, "skills": skills,
                "final_predicates": final_predicates,
                "reset_snapshot_sha256": reset_hash,
                "canonical_plan": strict_plan,
            })
            trace.episode_end(terminal="plan_consumed", planner_calls=planner.calls,
                              steps_used=steps_used, task_success=official)
        summary["video_frames"] = env.save_video(seed_dir / "episode.mp4")
        summary["video"] = "episode.mp4"
    except Exception as exc:  # preserve denominator and continue to the next fixed seed
        observed_steps = int(getattr(env, "_step_count", 0)) if env is not None else 0
        summary.update({
            "terminal": "runtime_error", "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(), "planner_calls": planner.calls,
            "steps_used": observed_steps,
        })
        try:
            if env is not None and env.frames:
                summary["video_frames"] = env.save_video(seed_dir / "episode.mp4")
                summary["video"] = "episode.mp4"
        except Exception as video_exc:
            summary["video_error"] = f"{type(video_exc).__name__}: {video_exc}"
    finally:
        trace.close()
        if env is not None:
            env.close()

    stage, reason = failure_from(summary)
    summary.update({
        "failure_stage": stage, "failure_reason": reason,
        "planner": {
            "calls": measured_service.calls,
            "latency_ms": measured_service.latencies_ms[0] if measured_service.latencies_ms else None,
            "raw_json": measured_service.raw_outputs[0] if measured_service.raw_outputs else None,
            "valid": summary.get("terminal") != "plan_rejected" and measured_service.calls == 1,
        },
        "mechanics": {
            "runtime_boundary_planner_calls": 0,
            "fallback_calls": 0,
            "repair_calls": 0,
            "sequence_corrections": 0,
            "action_before_plan": measured_policy.action_before_plan,
            "nan_violations": measured_policy.nan_violations,
        },
        "policy": {
            "calls": measured_policy.calls,
            "actions": int(summary.get("steps_used", 0)),
            "latencies_ms": measured_policy.latencies_ms,
            "p50_ms": percentile(measured_policy.latencies_ms, 0.5),
            "p95_ms": percentile(measured_policy.latencies_ms, 0.95),
            "max_ms": max(measured_policy.latencies_ms) if measured_policy.latencies_ms else None,
            "deadline_misses": sum(v > CHUNK_DEADLINE_MS for v in measured_policy.latencies_ms),
        },
        "elapsed_s": time.monotonic() - started,
    })
    (seed_dir / "result.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({
        "seed": seed, "planner_valid": summary["planner"]["valid"],
        "skills": [f"{s.get('skill')}:{s.get('status')}" for s in summary.get("skills", [])],
        "actions": summary["policy"]["actions"], "official": summary.get("task_success"),
        "failure_stage": stage,
    }), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--server-addr", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=10086)
    ap.add_argument("--model-path", default="/checkpoint")
    ap.add_argument("--split", default="target")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--episode-budget", type=int, default=600)
    ap.add_argument("--planner-timeout", type=float, default=10.0)
    ap.add_argument("--video-stride", type=int, default=2)
    ap.add_argument("--video-fps", type=int, default=20)
    ap.add_argument("--verifier-config", type=Path, required=True)
    ap.add_argument("--sync-verifier", action="store_true")
    ap.add_argument("--no-event-gate", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    if seeds != SEEDS:
        ap.error("canonical measurement requires exactly seeds 5000..5019 in order")
    if args.source_commit != "45280448bd9f650c3cfd94f26939a8830374a05b":
        ap.error("source commit must be the fixed parent commit 45280448bd9f650c3cfd94f26939a8830374a05b")
    operating_points = json.loads(args.verifier_config.read_text())["runtime_operating_points"]
    registry = SkillRegistry(bindings.CONTRACTS)
    base_policy = BasePolicyClient(
        args.model_path, args.server_addr, args.server_port, "robocasa365", 0.95,
        POLICY_CHUNK_ACTIONS, AdapterMode.DISABLED)
    backend = RemoteVLMScorerBackend(
        base_policy.processor(), args.server_addr, args.server_port, robot_type="robocasa365")
    config = {
        "task": "CloseBlenderLid", "seeds": list(seeds), "split": args.split,
        "source_commit": args.source_commit, "runtime_contract": runtime_contract_dict(),
        "planner": {"kind": PreEpisodePlanner.kind, "calls_per_episode": 1,
                    "boundary_calls": 0, "repair": False, "fallback": False},
        "verifier": {"kind": "obs_vlm", "simulator_gt_input": False,
                     "config": str(args.verifier_config)},
        "policy": base_policy.provenance(), "baseline_official_success": "3/20",
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    episodes = []
    try:
        for seed in seeds:
            episodes.append(run_seed(args, registry, base_policy, backend, operating_points, seed))
    finally:
        backend.close()
        base_policy.close()

    rows = []
    for episode in episodes:
        by_skill = {row["skill"]: row for row in episode.get("skills", [])}
        rows.append({
            "seed": episode["seed"],
            "reset_snapshot_sha256": episode.get("reset_snapshot_sha256"),
            "planner_valid": episode["planner"]["valid"],
            "planner_calls": episode["planner"]["calls"],
            "planner_latency_ms": episode["planner"]["latency_ms"],
            "grasp_reached": "GRASP_OBJECT" in by_skill,
            "grasp_status": by_skill.get("GRASP_OBJECT", {}).get("status"),
            "move_reached": "MOVE_OBJECT" in by_skill,
            "move_status": by_skill.get("MOVE_OBJECT", {}).get("status"),
            "place_reached": "PLACE_OBJECT" in by_skill,
            "place_status": by_skill.get("PLACE_OBJECT", {}).get("status"),
            "actions": episode["policy"]["actions"],
            "policy_calls": episode["policy"]["calls"],
            "policy_p95_ms": episode["policy"]["p95_ms"],
            "official_success": bool(episode.get("task_success")),
            "failure_stage": episode.get("failure_stage"),
            "failure_reason": episode.get("failure_reason"),
        })
    with (args.out / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "episodes.json").write_text(json.dumps(episodes, indent=2, default=str) + "\n")
    print(json.dumps({"completed": len(episodes),
                      "planner_valid": sum(row["planner_valid"] for row in rows),
                      "official_success": sum(row["official_success"] for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
