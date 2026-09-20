#!/usr/bin/env python3
"""Measure CloseBlenderLid reset and camera-producing env.step throughput."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

for _path in ("/train/src", "/work", "/skill_eval_tools", "/rl_env/src"):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import rollout
from grpo_train_loop import _make_env
from robocasa.utils.env_utils import convert_action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split", default="pretrain")
    parser.add_argument("--replan-steps", type=int, default=16)
    parser.add_argument("--output")
    args = parser.parse_args()

    create_started = time.perf_counter()
    env, _ = _make_env(args.split, args.seed)
    create_seconds = time.perf_counter() - create_started
    try:
        reset_started = time.perf_counter()
        obs, _ = rollout.reset_env(env, args.seed)
        reset_seconds = time.perf_counter() - reset_started
        camera_shapes = {key: list(np.asarray(obs[key]).shape) for key in rollout.CAMERA_KEYS}
        action = convert_action(np.zeros(rollout.ACTION_DIM, dtype=np.float32))
        for _ in range(args.warmup):
            obs, _, done, truncated, _ = env.step(action)
            if done or truncated:
                obs, _ = rollout.reset_env(env, args.seed)
        started = time.perf_counter()
        resets = 0
        for index in range(args.steps):
            obs, _, done, truncated, _ = env.step(action)
            if done or truncated:
                resets += 1
                obs, _ = rollout.reset_env(env, args.seed + index + 1)
        wall = time.perf_counter() - started
    finally:
        env.close()

    step_fps = args.steps / wall
    report = {
        "steps": args.steps,
        "create_seconds": create_seconds,
        "reset_seconds": reset_seconds,
        "step_wall_seconds": wall,
        "steps_per_second": step_fps,
        "estimated_policy_requests_per_second_per_worker": step_fps / args.replan_steps,
        "replan_steps": args.replan_steps,
        "resets_during_measurement": resets,
        "camera_shapes": camera_shapes,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
