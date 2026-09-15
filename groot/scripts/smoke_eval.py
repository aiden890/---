"""Smoke test for the GR00T eval image: verify imports + the robocasa/CloseBlenderLid
gym env registers and renders under EGL, WITHOUT loading the policy or hitting HF.
Run inside the container with the asset volume mounted."""
import os
import numpy as np

print("[smoke] importing gr00t / robocasa / robosuite ...")
import gr00t  # noqa: F401
import robocasa  # noqa: F401
import robosuite  # noqa: F401
import gymnasium as gym
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon

# Import the gym_wrapper so robocasa/<Task> ids register.
import robocasa.wrappers.gym_wrapper  # noqa: F401

assert "panda_omron" in DATA_CONFIG_MAP, "panda_omron data_config missing"
assert "CloseBlenderLid" in TASK_SET_REGISTRY["atomic_seen"], "CloseBlenderLid not in atomic_seen"
horizon = get_task_horizon("CloseBlenderLid")
print(f"[smoke] CloseBlenderLid horizon={horizon}, atomic_seen size={len(TASK_SET_REGISTRY['atomic_seen'])}")

print("[smoke] gym.make robocasa/CloseBlenderLid (split=pretrain, enable_render=True) ...")
env = gym.make("robocasa/CloseBlenderLid", split="pretrain", enable_render=True)
obs, info = env.reset()
print("[smoke] reset ok. obs keys:", sorted(list(obs.keys()))[:12])
frame = env.render()
print("[smoke] render frame:", None if frame is None else (np.asarray(frame).shape, np.asarray(frame).dtype))
# one no-op-ish step to confirm the action space and predicate/success plumbing
act = env.action_space.sample()
obs, rew, done, trunc, info = env.step(act)
print("[smoke] step ok. reward=", rew, "done=", done, "success=", info.get("success"))
env.close()
print("[smoke] OK")
