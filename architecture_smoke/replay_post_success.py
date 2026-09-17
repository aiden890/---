"""Replay labeled skill actions and append a real simulator post-success dwell.

The original balanced corpus stops at the first GT-positive frame, which makes
held-out online-boundary recall mathematically capped below 0.5.  This utility
replays the recorded (observation-only policy) actions from the saved initial
state, verifies that the same GT boundary is reached, then executes stationary
hold actions for N steps while recording fresh camera/proprio observations.
Simulator predicates remain offline labels only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, "/work")
import rollout

import gymnasium as gym
import robocasa  # noqa: F401
from robocasa.utils.env_utils import convert_action
from robosuite.utils.binding_utils import MjSimState

from skill_eval import Sim, f, hold_action


def load_action_steps(path):
    return [row for row in (json.loads(line) for line in open(path))
            if row.get("type") == "step"]


def restore_saved_start(sim, npz_path):
    snap = np.load(npz_path)
    sim.k.sim.set_state(MjSimState(float(snap["time"]), snap["qpos"].copy(),
                                  snap["qvel"].copy()))
    sim.k.sim.forward()
    sim.gripper.current_action = snap["gripper_current_action"].copy()
    blender = sim.k.blender
    blender._lid_on_blender = False
    blender._turned_on = False
    blender._button_contact_prev_timestep = False
    sim.robot.composite_controller.update_state()
    sim.robot.composite_controller.reset()
    sim.k.timestep = 0
    sim.k.cur_time = float(snap["time"])
    sim.k.done = False


def skill_success(skill, predicates):
    if skill == "grasp":
        return predicates["lid_grasped"]
    if skill == "move_holding":
        return predicates["lid_grasped"] and predicates["in_preplace_region"]
    if skill == "place":
        return predicates["official_check_success"]
    raise ValueError(skill)


def replay_one(source, output, skill, seed, dwell, video_stride, video_fps):
    source_seed = source / f"seed{seed}"
    result = json.loads((source_seed / f"{skill}.json").read_text())
    if not result.get("success"):
        raise ValueError(f"source {skill}/seed{seed} is not a labeled success")
    source_success_step = int(result["success_step"])
    actions = load_action_steps(source_seed / f"{skill}_steps.jsonl")
    if len(actions) < source_success_step:
        raise ValueError(f"source action log ends before success: {len(actions)} < {source_success_step}")

    env = gym.make("robocasa/CloseBlenderLid", split="pretrain", seed=seed)
    sim = Sim(env)
    try:
        obs, _ = rollout.reset_env(env, seed)
        scene = json.loads((source_seed / "scene.json").read_text())
        sim.rest_lid_pos = np.asarray(scene["rest_lid_pos"], dtype=float)
        sim.rest_xy_to_closed = float(scene["rest_xy_to_closed"])
        if skill != "grasp":
            restore_saved_start(sim, source_seed / f"{skill}_start_state.npz")
            obs, _, _, _, _ = env.step(convert_action(hold_action(True)))

        output.mkdir(parents=True, exist_ok=True)
        frames = [rollout.make_video_frame(obs)]
        log_path = output / f"{skill}_steps.jsonl"
        reached = None
        with open(log_path, "w") as log:
            log.write(json.dumps({"type": "start", "kind": "replay_post_success",
                                  "skill": skill, "source": str(source_seed),
                                  "post_success_hold": dwell}) + "\n")
            for index, source_row in enumerate(actions[:source_success_step], start=1):
                action = np.asarray(source_row["action"], dtype=np.float32)
                obs, _, done, trunc, info = env.step(convert_action(action))
                predicates = sim.predicates()
                state = rollout.observation_to_state(obs)
                if reached is None and skill_success(skill, predicates):
                    # Keep the source label contract. GRASP/MOVE predicates have
                    # temporal holds in the source evaluator, so the authoritative
                    # boundary remains source_success_step below.
                    reached = index
                row = {"type": "step", "step": index, "action": f(action),
                       "proprio": f(state), "predicates": predicates,
                       "official_success": bool(info.get("success", False)),
                       "success_step": source_success_step if index >= source_success_step else None,
                       "phase": "post_success" if index >= source_success_step else "pre_success"}
                log.write(json.dumps(row) + "\n")
                if index % video_stride == 0 or index == source_success_step or done or trunc:
                    frames.append(rollout.make_video_frame(obs))
                    log.write(json.dumps({"type": "frame", "step": index,
                                          "frame_index": len(frames) - 1}) + "\n")
                if done or trunc:
                    raise RuntimeError(f"environment ended during replay at step {index}")
            boundary_predicates = sim.predicates()
            if not skill_success(skill, boundary_predicates):
                raise RuntimeError(
                    f"replay did not reproduce {skill}/seed{seed} boundary at "
                    f"step {source_success_step}: {boundary_predicates}")
            log.write(json.dumps({"type": "success", "step": source_success_step,
                                  "success_step": source_success_step}) + "\n")
            closed = skill in ("grasp", "move_holding")
            extra = 0
            for extra in range(1, dwell + 1):
                step = source_success_step + extra
                action = hold_action(closed)
                obs, _, done, trunc, info = env.step(convert_action(action))
                predicates = sim.predicates()
                state = rollout.observation_to_state(obs)
                log.write(json.dumps({"type": "step", "step": step, "action": f(action),
                                      "proprio": f(state), "predicates": predicates,
                                      "official_success": bool(info.get("success", False)),
                                      "success_step": source_success_step,
                                      "phase": "post_success"}) + "\n")
                if step % video_stride == 0 or extra == dwell or done or trunc:
                    frames.append(rollout.make_video_frame(obs))
                    log.write(json.dumps({"type": "frame", "step": step,
                                          "frame_index": len(frames) - 1}) + "\n")
                if done or trunc:
                    break
        imageio.mimsave(output / f"{skill}.mp4", frames, fps=video_fps)
        summary = {"skill": skill, "seed": seed, "success": True,
                   "success_step": source_success_step, "post_success_steps": extra,
                   "terminated_by": "replay_post_success_hold", "video_frames": len(frames),
                   "source": str(source_seed)}
        (output / f"{skill}.json").write_text(json.dumps(summary, indent=2))
        (output / "results.json").write_text(json.dumps({skill: summary}, indent=2))
        (output / "scene.json").write_text(json.dumps(scene, indent=2))
        print(json.dumps(summary), flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--skill", required=True, choices=("grasp", "move_holding", "place"))
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--dwell", type=int, default=32)
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=20)
    args = parser.parse_args()
    for seed in (int(value) for value in args.seeds.split(",")):
        replay_one(Path(args.source), Path(args.out) / args.skill / f"seed{seed}",
                   args.skill, seed, args.dwell, args.video_stride, args.video_fps)


if __name__ == "__main__":
    main()
