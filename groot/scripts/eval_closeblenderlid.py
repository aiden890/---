"""Single-task GR00T N1.5 eval driver for CloseBlenderLid (RoboCasa365).

Reuses the robocasa-benchmark/Isaac-GR00T inference stack verbatim (Gr00tPolicy +
RobotInferenceServer + SimulationInferenceClient) — we only narrow run_eval.py's
task_set loop to one env so we get a clean apples-to-apples number vs our Xiaomi
CloseBlenderLid result, plus per-episode success + saved MP4s.

Server + client run in one process on one GPU (threaded), same as run_eval.py's
combined mode. split defaults to `target` to match Xiaomi's RoboCasa365 target50 protocol.
"""
import argparse
import json
import os
import threading
import time

import numpy as np
from robocasa.utils.dataset_registry_utils import get_task_horizon

from gr00t.eval.robot import RobotInferenceServer
from gr00t.eval.simulation import (
    MultiStepConfig,
    SimulationConfig,
    SimulationInferenceClient,
    VideoConfig,
)
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


def run_server(data_config, model_path, embodiment_tag, port):
    dc = DATA_CONFIG_MAP[data_config]
    policy = Gr00tPolicy(
        model_path=model_path,
        modality_config=dc.modality_config(),
        modality_transform=dc.transform(),
        embodiment_tag=embodiment_tag,
        denoising_steps=4,
    )
    RobotInferenceServer(policy, port=port).run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--embodiment_tag", default="new_embodiment")
    ap.add_argument("--data_config", default="panda_omron")
    ap.add_argument("--task", default="CloseBlenderLid")
    ap.add_argument("--split", default="target", choices=["pretrain", "target"])
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--n_envs", type=int, default=1)
    ap.add_argument("--n_action_steps", type=int, default=16)
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--video_dir", required=True)
    args = ap.parse_args()

    # Server in a daemon thread; client in main thread (single GPU).
    threading.Thread(
        target=run_server,
        args=(args.data_config, args.model_path, args.embodiment_tag, args.port),
        daemon=True,
    ).start()
    time.sleep(3)

    client = SimulationInferenceClient(host="localhost", port=args.port)
    print("modality:", list(client.get_modality_config().keys()), flush=True)

    horizon = get_task_horizon(args.task)
    this_video_dir = os.path.join(args.video_dir, "evals", args.split, args.task)
    os.makedirs(this_video_dir, exist_ok=True)
    print(f"[eval] task={args.task} split={args.split} n_ep={args.n_episodes} "
          f"horizon={horizon} n_action_steps={args.n_action_steps}", flush=True)

    config = SimulationConfig(
        env_name=f"robocasa/{args.task}",
        split=args.split,
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
        video=VideoConfig(video_dir=this_video_dir),
        multistep=MultiStepConfig(n_action_steps=args.n_action_steps, max_episode_steps=horizon),
    )

    t0 = time.time()
    env_name, episode_successes = client.run_simulation(config)
    dt = time.time() - t0

    successes = [bool(x) for x in episode_successes]
    sr = float(np.mean(successes)) if successes else 0.0
    stats = {
        "task": args.task,
        "split": args.split,
        "model_path": args.model_path,
        "embodiment_tag": args.embodiment_tag,
        "data_config": args.data_config,
        "n_action_steps": args.n_action_steps,
        "horizon": horizon,
        "num_episodes": len(successes),
        "num_success": int(sum(successes)),
        "success_rate": sr,
        "episode_successes": successes,
        "wall_seconds": round(dt, 1),
    }
    stats_path = os.path.join(this_video_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[eval] SUCCESS_RATE={sr:.4f}  ({stats['num_success']}/{stats['num_episodes']})  "
          f"{dt:.0f}s", flush=True)
    print(f"[eval] stats -> {stats_path}", flush=True)
    print(f"[eval] videos -> {this_video_dir}", flush=True)


if __name__ == "__main__":
    main()
