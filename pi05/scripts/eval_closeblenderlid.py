#!/usr/bin/env python3
"""pi0.5 (openpi) single-task RoboCasa eval CLIENT for CloseBlenderLid.

Apples-to-apples with GR00T (t_ff372eea) and Xiaomi: same task, same split,
same PandaOmron embodiment, same official success predicate (env info["success"]).

This is the CLIENT half of the openpi 2-part protocol. It talks over websocket to
`scripts/serve_policy.py` (the JAX pi0.5 server, separate process/image). The obs->
model mapping, resize (180deg via resize_with_pad in openpi image_tools), replan and
`convert_action` are copied verbatim from the vendor eval harness
(robocasa-benchmark/openpi examples/robocasa/main.py @ 5a6beda9) so numbers match the
official recipe; we only narrow it to ONE task and add per-step jsonl + SHA256 so the
output parallels our skill_eval artifacts.

Runs inside the reused groot-eval image (has robocasa/robosuite/EGL + openpi_client's
light deps). No JAX here — inference is remote.
"""
import argparse
import collections
import hashlib
import json
import logging
import os
import pathlib
from datetime import datetime

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wcp

import gymnasium as gym
import robocasa  # noqa: F401  (registers robocasa/<Task> gym ids)
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_state(obs) -> np.ndarray:
    # identical concatenation order to vendor main.py
    return np.concatenate(
        (
            obs["state.end_effector_position_relative"],
            obs["state.end_effector_rotation_relative"],
            obs["state.base_position"],
            obs["state.base_rotation"],
            obs["state.gripper_qpos"],
        ),
        axis=0,
    )


def run(args):
    logging.basicConfig(level=logging.INFO)
    np.random.seed(args.seed)
    task = args.task
    split = args.split
    horizon = get_task_horizon(task)
    resize = args.resize_size
    replan = args.replan_steps

    out = pathlib.Path(args.video_dir)
    out.mkdir(parents=True, exist_ok=True)

    client = _wcp.WebsocketClientPolicy(args.host, args.port)
    logging.info("connected to policy server %s:%d", args.host, args.port)

    env = gym.make(f"robocasa/{task}", split=split, seed=args.seed)
    inner = getattr(env.unwrapped, "env", env.unwrapped)

    per_episode = []
    total_succ = 0
    for ep in range(args.n_episodes):
        obs, info = env.reset()
        task_lang = obs["annotation.human.task_description"]
        action_plan = collections.deque()
        replay_images = []
        steps_log = []  # per-step predicate/eef timeline (parity w/ skill_eval)
        success_step = None
        t = 0
        done = False
        while t < horizon:
            img = np.ascontiguousarray(obs["video.robot0_agentview_left"])
            wrist_img = np.ascontiguousarray(obs["video.robot0_eye_in_hand"])
            img_right = np.ascontiguousarray(obs["video.robot0_agentview_right"])
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize, resize))
            wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize, resize))
            img_right = image_tools.convert_to_uint8(image_tools.resize_with_pad(img_right, resize, resize))

            if not action_plan:
                state = build_state(obs)
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/right_image": img_right,
                    "observation/state": state,
                    "prompt": task_lang,
                }
                action_chunk = client.infer(element)["actions"]
                assert len(action_chunk) >= replan, (
                    f"want replan={replan}, policy predicts {len(action_chunk)}"
                )
                action_plan.extend(action_chunk[:replan])

            action = convert_action(action_plan.popleft())
            obs, reward, done, truncated, info = env.step(action)
            done = bool(info["success"])  # robocasa: official success predicate

            replay_img = np.ascontiguousarray(env.render())
            replay_img = image_tools.convert_to_uint8(replay_img)
            if t % 2 == 0 or t == horizon - 1 or done:
                replay_images.append(replay_img)

            # per-step timeline: official success + eef pos (for parity/inspection)
            try:
                eef = np.asarray(obs["state.end_effector_position_relative"]).tolist()
            except Exception:
                eef = None
            steps_log.append({"type": "step", "t": t, "success": done, "eef_rel": eef})

            if done and success_step is None:
                success_step = t
            if done or truncated:
                break
            t += 1

        if done:
            total_succ += 1
        suffix = "success" if done else "failure"
        vid = out / f"{task}_ep{ep:02d}_{suffix}.mp4"
        imageio.mimwrite(vid, [np.asarray(x) for x in replay_images], fps=20)
        # jsonl: header + per-step + frame index rows
        jl = out / f"{task}_ep{ep:02d}_steps.jsonl"
        with open(jl, "w") as f:
            f.write(json.dumps({
                "type": "meta", "task": task, "split": split, "episode": ep,
                "seed": args.seed, "horizon": horizon, "replan_steps": replan,
                "success": bool(done), "success_step": success_step,
                "prompt": task_lang, "video": vid.name, "video_fps": 20,
                "video_stride": 2,
            }) + "\n")
            for row in steps_log:
                f.write(json.dumps(row) + "\n")
        per_episode.append({
            "episode": ep, "success": bool(done), "success_step": success_step,
            "n_steps": t + 1, "video": vid.name,
            "video_sha256": sha256_file(vid), "steps_sha256": sha256_file(jl),
        })
        logging.info("ep %d: %s (%d/%d so far)", ep, suffix, total_succ, ep + 1)

    stats = {
        "task": task, "split": split, "model": "pi05_pretrain_human300/multitask_learning/75000",
        "vendor": "robocasa-benchmark/openpi@5a6beda9", "embodiment": "panda_omron",
        "n_episodes": args.n_episodes, "n_successes": total_succ,
        "success_rate": total_succ / max(1, args.n_episodes),
        "replan_steps": replan, "seed": args.seed, "horizon": horizon,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "episodes": per_episode,
    }
    sp = out / f"stats_{task}_{split}.json"
    with open(sp, "w") as f:
        json.dump(stats, f, indent=2)
    print("SUCCESS_RATE", f"{total_succ}/{args.n_episodes}={stats['success_rate']:.4f}")
    print("STATS", sp)
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="CloseBlenderLid")
    ap.add_argument("--split", default="pretrain", choices=["pretrain", "target"])
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--replan_steps", type=int, default=16)  # match Xiaomi/GR00T replan16
    ap.add_argument("--resize_size", type=int, default=224)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--video_dir", default="/output/CloseBlenderLid")
    run(ap.parse_args())
