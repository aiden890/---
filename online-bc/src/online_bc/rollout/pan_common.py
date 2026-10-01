"""Shared, read-only PanTransfer pouring probe; no optimizer or weight updates."""

import collections
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

sys.path[:0] = ["/train/src", "/work", "/rl_env/src"]
import rollout  # noqa: E402 -- simulator path bootstrap
from robocasa.utils import object_utils as OU  # noqa: E402 -- simulator path bootstrap
from robosuite.utils.binding_utils import MjSimState  # noqa: E402 -- simulator path bootstrap

SKILL = "Tilt the pan you are holding to pour the vegetables onto the plate. Keep holding the pan and do not touch the vegetables with your hand."
FULL = "Pick up the pan and dump the vegetables in it onto the plate. Then return the pan to the stove."


def queues(obs, history=None):
    if history is None:
        history = dict(
            images={k: [v] for k, v in rollout.collect_images(obs).items()},
            states=[rollout.observation_to_state(obs)],
        )
    return {
        k: collections.deque(v, maxlen=7) for k, v in history["images"].items()
    }, collections.deque(history["states"], maxlen=7)


def append(obs, images, states):
    for k, v in rollout.collect_images(obs).items():
        images[k].append(v)
    states.append(rollout.observation_to_state(obs))


def state_hash(k):
    return hashlib.sha256(k.sim.data.qpos.tobytes() + k.sim.data.qvel.tobytes()).hexdigest()


def predicates(k, rest_z):
    pan = k.objects["vegetable_container"]
    plate = k.objects["plate"]
    g = k.robots[0].gripper["right"]
    pos = np.array(k.sim.data.body_xpos[k.obj_body_id[pan.name]])
    pp = np.array(k.sim.data.body_xpos[k.obj_body_id[plate.name]])
    mat = np.array(k.sim.data.body_xmat[k.obj_body_id[pan.name]]).reshape(3, 3)
    touched = bool(k._robot_touched_food)
    contact = bool(k.check_contact(g, pan))
    return dict(
        pan_grasped=bool(contact and pos[2] - rest_z > 0.04),
        bilateral_geom_grasp=bool(k._check_grasp(g, pan)),
        pan_contact=contact,
        pan_lift=float(pos[2] - rest_z),
        pan_plate_xy=float(np.linalg.norm(pos[:2] - pp[:2])),
        pan_plate_dz=float(pos[2] - pp[2]),
        pan_tilt_deg=float(np.degrees(np.arccos(np.clip(mat[2, 2], -1, 1)))),
        food_in_pan=bool(OU.check_obj_in_receptacle(k, "vegetable", "vegetable_container")),
        food_on_plate=bool(OU.check_obj_in_receptacle(k, "vegetable", "plate")),
        robot_touched_food=touched,
        pour_success=bool(OU.check_obj_in_receptacle(k, "vegetable", "plate") and not touched),
        full_task_success=bool(k._check_success()),
    )


def eligible(p):
    return (
        p["pan_grasped"]
        and p["pan_lift"] > 0.04
        and p["pan_plate_xy"] < 0.35
        and p["food_in_pan"]
        and not p["robot_touched_food"]
        and not p["food_on_plate"]
    )


def capture(env, obs, images, states, actions, seed, rest_z):
    k = env.unwrapped.env
    return dict(
        seed=seed,
        rest_pan_z=rest_z,
        prefix_steps=len(actions),
        prefix_actions=np.array(actions, np.float32),
        snapshot={
            name: np.array(getattr(k.sim.data, name), copy=True)
            for name in [
                "qpos",
                "qvel",
                "qacc_warmstart",
                "ctrl",
                "act",
                "qfrc_applied",
                "xfrc_applied",
            ]
        },
        time=float(k.sim.data.time),
        gripper_current_action=np.array(k.robots[0].gripper["right"].current_action, copy=True),
        robot_touched_food=bool(k._robot_touched_food),
        observation=obs,
        history=dict(images={key: list(v) for key, v in images.items()}, states=list(states)),
        state_hash=state_hash(k),
        model_xml=k.sim.model.get_xml(),
        ep_meta=k.get_ep_meta(),
        predicates=predicates(k, rest_z),
        criterion="pan grasped + lifted >4cm + pan/plate XY <35cm + food remains in pan + no hand/food contact; checked at 16-step boundary",
    )


def restore(env, b):
    k = env.unwrapped.env
    if k.sim.data.qpos.shape != b["snapshot"]["qpos"].shape:
        raise RuntimeError("Scene state dimensions differ")
    k.sim.set_state(
        MjSimState(b["time"], b["snapshot"]["qpos"].copy(), b["snapshot"]["qvel"].copy())
    )
    for key in ["qacc_warmstart", "ctrl", "act", "qfrc_applied", "xfrc_applied"]:
        getattr(k.sim.data, key)[:] = b["snapshot"][key]
    k.sim.forward()
    k.robots[0].gripper["right"].current_action = b["gripper_current_action"].copy()
    k._robot_touched_food = b["robot_touched_food"]
    k.robots[0].composite_controller.update_state()
    k.robots[0].composite_controller.reset()
    k.timestep = 0
    k.cur_time = b["time"]
    k.done = False
    assert state_hash(k) == b["state_hash"]
    assert eligible(predicates(k, b["rest_pan_z"]))
    return env.unwrapped.get_observation(k._get_observations(force_update=True))


def write(path, data):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)


def overlay(obs, label):
    from PIL import Image, ImageDraw

    im = Image.fromarray(rollout.make_video_frame(obs))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, im.width, 26), fill="#111827")
    d.text((5, 7), label, fill="white")
    return np.array(im)
