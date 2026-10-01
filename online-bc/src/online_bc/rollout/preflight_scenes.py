"""Reset real training scenes before beginning costly rollout collection."""

import sys
import json
import argparse
import random

sys.path.insert(0, "/results")
from online_bc.rollout.pan_common import rollout
import numpy as np
import gymnasium as gym
import robocasa  # noqa: F401 -- registers RoboCasa Gym environments

p = argparse.ArgumentParser()
p.add_argument("--seeds", required=True)
a = p.parse_args()
for seed in map(int, a.seeds.split(",")):
    np.random.seed(seed)
    random.seed(seed)
    env = gym.make("robocasa/PrepareCoffee", split="pretrain", seed=seed)
    try:
        o, _ = rollout.reset_env(env, seed)
        assert all(k in o for k in rollout.CAMERA_KEYS)
        print(
            json.dumps(
                dict(
                    seed=seed, passed=True, instruction=str(o["annotation.human.task_description"])
                )
            ),
            flush=True,
        )
    finally:
        env.close()
