#!/usr/bin/env python3
"""Compare per-member reset with one-reset snapshot restore for a GRPO group."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

for _path in ("/train/src", "/work", "/skill_eval_tools", "/rl_env/src"):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import rollout
from grpo_train_loop import TrainerClient, _current_obs, _make_env, _run_one_skill
from reward import RewardConfig, RewardManager
from skill_manager import Skill


def run_group(strategy: str, env, sim, client, args, round_index: int) -> dict:
    metrics = client.metrics()
    policy_version = int(metrics["policy_version"])
    reset_times: list[float] = []
    restore_times: list[float] = []
    rollout_times: list[float] = []
    trajectory_ids: list[str] = []
    group_started = time.perf_counter()
    snapshot = None

    if strategy == "snapshot":
        started = time.perf_counter()
        rollout.reset_env(env, args.seed)
        reset_times.append(time.perf_counter() - started)
        sim.rest_lid_pos = sim.lid_pos()
        snapshot = sim.snapshot(f"bench-group-{round_index}", 0)

    try:
        for member in range(args.group_size):
            started = time.perf_counter()
            if strategy == "reset":
                rollout.reset_env(env, args.seed)
                reset_times.append(time.perf_counter() - started)
                sim.rest_lid_pos = sim.lid_pos()
            else:
                sim.restore(snapshot)
                restore_times.append(time.perf_counter() - started)

            # Both paths take the same single hold step so their policy observation starts
            # from an equivalent settled state.
            obs = _current_obs(sim)
            traj_id = f"reuse-bench-{round_index}-{strategy}-m{member}"
            trajectory_ids.append(traj_id)
            started = time.perf_counter()
            _run_one_skill(
                sim, client, obs, args, Skill.GRASP,
                RewardManager(RewardConfig(horizon=args.horizon, use_milestones=False)),
                eta=args.eta, traj_id=traj_id, seed=args.seed * 1000 + member,
                expected_policy_version=policy_version,
            )
            rollout_times.append(time.perf_counter() - started)
    finally:
        client.discard_store(trajectory_ids)

    wall = time.perf_counter() - group_started
    return {
        "strategy": strategy,
        "group_size": args.group_size,
        "group_wall_seconds": wall,
        "members_per_second": args.group_size / wall,
        "reset_count": len(reset_times),
        "reset_seconds_total": sum(reset_times),
        "restore_count": len(restore_times),
        "restore_seconds_total": sum(restore_times),
        "rollout_seconds_total": sum(rollout_times),
        "rollout_seconds_mean": statistics.mean(rollout_times),
        "policy_version": policy_version,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10088)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--replan-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=710000)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--output")
    args = parser.parse_args()
    args.split = "pretrain"
    args.horizon_grasp = args.horizon_move = args.horizon_place = args.horizon
    args.obs_history = 4
    args.obs_interval = 2
    args.video_stride = 2
    args.video_fps = 20
    args.reward_variant = "simulator_terminal_only"
    args.skill_success_reward = 1.0
    args.skill_success_gamma = 0.998
    args.skill_success_decay = True

    client = TrainerClient(args.model_path, args.host, args.port, "robocasa365", 0.95)
    env, sim = _make_env(args.split, args.seed)
    rows = []
    try:
        for round_index in range(args.rounds):
            order = ("reset", "snapshot") if round_index % 2 == 0 else ("snapshot", "reset")
            for strategy in order:
                rows.append(run_group(strategy, env, sim, client, args, round_index))
    finally:
        env.close()
        client.close()

    means = {
        strategy: statistics.mean(
            row["group_wall_seconds"] for row in rows if row["strategy"] == strategy)
        for strategy in ("reset", "snapshot")
    }
    report = {
        "group_size": args.group_size,
        "rounds": args.rounds,
        "horizon": args.horizon,
        "replan_steps": args.replan_steps,
        "mean_group_wall_seconds": means,
        "snapshot_speedup": means["reset"] / means["snapshot"],
        "rows": rows,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
