"""run_architecture_smoke.py -- thin composition / CLI entry point.

Adapter-free full-architecture inference smoke test for CloseBlenderLid:

    Planner (oracle stub) -> SkillCall -> ExecutionManager -> base Xiaomi
    RoboCasa365 VLA (adapter DISABLED) -> RoboCasa -> PredicateVerifier ->
    SkillResult -> Planner,  for N randomized seeds.

This module only wires layers together and writes artifacts; all behavior lives
in the imported modules. Run inside the xiaomi-client container (rollout.py +
skill_eval.py on the path, VLA server reachable). See README.md for the data
flow and the one launch command.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from schemas import AdapterMode
import bindings
from skills import SkillRegistry
from planner import OraclePlanner
from environment import RoboCasaEnvironment
from executor import ExecutionManager
from trace import Trace
from episode import run_episode


def build_registry() -> SkillRegistry:
    return SkillRegistry(bindings.CONTRACTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--split", default="pretrain")
    ap.add_argument("--server-addr", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=10086)
    ap.add_argument("--model-path", default="/checkpoint")
    ap.add_argument("--robot-type", default="robocasa365")
    ap.add_argument("--replan-steps", type=int, default=16)
    ap.add_argument("--obs-history", type=int, default=4)
    ap.add_argument("--obs-interval", type=int, default=2)
    ap.add_argument("--crop-ratio", type=float, default=0.95)
    ap.add_argument("--video-stride", type=int, default=2)
    ap.add_argument("--video-fps", type=int, default=20)
    ap.add_argument("--episode-budget", type=int, default=600)
    ap.add_argument("--max-planner-calls", type=int, default=12)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]

    # lazy heavy import: build one base policy client, reuse across seeds.
    from policy import BasePolicyClient
    registry = build_registry()
    planner = OraclePlanner(registry)
    policy = BasePolicyClient(
        model_path=args.model_path, server_addr=args.server_addr, server_port=args.server_port,
        robot_type=args.robot_type, crop_ratio=args.crop_ratio, replan_steps=args.replan_steps,
        adapter_mode=AdapterMode.DISABLED,
    )
    provenance = policy.provenance()
    # adapter-disabled assertion, demonstrably logged.
    assert provenance["adapter_checkpoint"] is None
    assert provenance["adapter_mode"] in ("disabled", "base_only")

    config = {
        "task": "CloseBlenderLid", "goal": bindings.GOAL, "seeds": seeds, "split": args.split,
        "planner": {"kind": planner.kind, "note": "scripted oracle stub, NOT a learned VLM planner"},
        "adapter_mode": AdapterMode.DISABLED.value, "adapter_checkpoint": None,
        "policy_provenance": provenance,
        "replan_steps": args.replan_steps, "obs_history": args.obs_history,
        "obs_interval": args.obs_interval, "crop_ratio": args.crop_ratio,
        "video_stride": args.video_stride, "video_fps": args.video_fps,
        "episode_budget": args.episode_budget, "skill_catalog": registry.catalog(),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))

    episodes = []
    try:
        for seed in seeds:
            seed_dir = out / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            env = RoboCasaEnvironment(
                seed=seed, split=args.split, replan_steps=args.replan_steps,
                obs_history=args.obs_history, obs_interval=args.obs_interval,
                crop_ratio=args.crop_ratio, video_stride=args.video_stride,
                video_fps=args.video_fps,
            )
            trace = Trace(seed_dir / "trace.jsonl", {
                "seed": seed, "task": "CloseBlenderLid", "adapter_mode": AdapterMode.DISABLED.value,
                "adapter_checkpoint": None, "planner_kind": planner.kind,
                "policy_provenance": provenance,
            })
            try:
                env.reset()
                scene = env.scene_meta()
                (seed_dir / "scene.json").write_text(json.dumps(scene, indent=2, default=str))
                manager = ExecutionManager(registry, policy, env, trace, AdapterMode.DISABLED)
                summary = run_episode(planner, manager, env, trace, registry,
                                      bindings.GOAL, args.episode_budget, args.max_planner_calls)
                nframes = env.save_video(seed_dir / "episode.mp4")
                summary["video"] = "episode.mp4"
                summary["video_frames"] = nframes
                (seed_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
                episodes.append(summary)
                print(json.dumps({"seed": seed, "terminal": summary["terminal"],
                                  "task_success": summary["task_success"],
                                  "planner_calls": summary["planner_calls"],
                                  "skills": [s["skill"] + ":" + s["status"] for s in summary["skills"]]}),
                      flush=True)
            finally:
                trace.close()
                env.close()
    finally:
        policy.close()

    agg = {
        "task": "CloseBlenderLid", "adapter_mode": AdapterMode.DISABLED.value,
        "adapter_checkpoint": None, "policy_provenance": provenance,
        "n_episodes": len(episodes), "seeds": seeds,
        "task_success_count": sum(1 for e in episodes if e["task_success"]),
        "episodes": episodes,
    }
    (out / "summary.json").write_text(json.dumps(agg, indent=2, default=str))
    print(json.dumps({"aggregate": {"n": len(episodes),
                                    "task_success": agg["task_success_count"]}}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
