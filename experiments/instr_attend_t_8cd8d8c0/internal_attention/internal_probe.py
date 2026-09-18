#!/usr/bin/env python3
"""Run 3 fixed observation phases x 4 instructions through the attention server."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/work")
sys.path.insert(0, "/skilltools")
sys.path.insert(0, "/tools")
import gymnasium as gym  # noqa: E402
import robocasa  # noqa: E402,F401
import rollout  # noqa: E402
import skill_eval  # noqa: E402
from instr_probe import build_snapshots, make_queues  # noqa: E402
from robocasa.utils.env_utils import convert_action  # noqa: E402
from skill_eval import FULL_INSTRUCTION, SKILLS, Sim, hold_action  # noqa: E402


INSTRUCTIONS = {
    "correct_full": FULL_INSTRUCTION,
    "skill_grasp": SKILLS["grasp"],
    "skill_move": SKILLS["move_holding"],
    "skill_place": SKILLS["place"],
}


def infer_recorded(client, state_queue, image_queues, args, instruction, state_tag, label):
    states = rollout.sample_history(state_queue, args.obs_history, args.obs_interval)
    images = {
        key: rollout.sample_history(queue, args.obs_history, args.obs_interval)
        for key, queue in image_queues.items()
    }
    state_history = np.asarray(states, dtype=np.float32)
    state = np.zeros((1, state_history.shape[0], rollout.STATE_DIM), dtype=np.float32)
    state[0, :, : state_history.shape[-1]] = state_history
    inputs = client.processor.apply_chat_template(
        client._build_messages(images, instruction),
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        do_resize=False,
        state=state,
        robot_type=client.robot_type,
    )
    request = dict(inputs)
    request["task_id"] = client.robot_type
    request["attention_meta"] = {
        "state_tag": state_tag,
        "label": label,
        "instruction": instruction,
        "num_steps": 5,
        "action_length": int(request["action_mask"].shape[1]),
    }
    actions = client.client(**request)
    return actions[0, :, : rollout.ACTION_DIM].float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--gen-attempts", type=int, default=6)
    parser.add_argument("--gen-horizon", type=int, default=900)
    parser.add_argument("--server-addr", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=10086)
    parser.add_argument("--model-path", default="/checkpoint")
    own, remaining = parser.parse_known_args()
    args = rollout.parse_args(remaining)
    rollout.validate_args(args)
    args.seed = own.seed
    args.gen_attempts = own.gen_attempts
    args.gen_horizon = own.gen_horizon
    out = Path(own.out)
    out.mkdir(parents=True, exist_ok=True)

    client = rollout.EvalClient(
        own.model_path, own.server_addr, own.server_port, args.robot_type, args.crop_ratio
    )
    env = gym.make("robocasa/CloseBlenderLid", split=args.split, seed=own.seed)
    sim = Sim(env)
    results = []
    try:
        observation, _ = rollout.reset_env(env, own.seed)
        if observation["annotation.human.task_description"] != FULL_INSTRUCTION:
            raise RuntimeError("unexpected reset instruction")
        sim.rest_lid_pos = sim.lid_pos()
        sim.rest_xy_to_closed = float(
            np.linalg.norm(sim.rest_lid_pos[:2] - sim.closed_pos()[:2])
        )
        snapshots = build_snapshots(client, sim, args, out)
        missing = sorted({"move", "place"} - snapshots.keys())
        if missing:
            raise RuntimeError(f"generation did not produce snapshots: {missing}")

        for state_tag in ("reset", "move", "place"):
            if state_tag == "reset":
                observation, _ = rollout.reset_env(env, own.seed)
            else:
                sim.restore(snapshots[state_tag])
                observation, _, _, _, _ = env.step(convert_action(hold_action(True)))
            predicates = sim.predicates()
            for label, instruction in INSTRUCTIONS.items():
                image_queues, state_queue = make_queues(
                    observation, args.obs_history, args.obs_interval
                )
                actions = infer_recorded(
                    client,
                    state_queue,
                    image_queues,
                    args,
                    instruction,
                    state_tag,
                    label,
                )
                row = {
                    "state_tag": state_tag,
                    "label": label,
                    "instruction": instruction,
                    "predicates": predicates,
                    "action_shape": list(actions.shape),
                    "first_action": actions[0].tolist(),
                }
                results.append(row)
                print(f"recorded {state_tag}/{label}: {actions.shape}", flush=True)
        (out / "probe_actions.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
    finally:
        env.close()
        client.close()


if __name__ == "__main__":
    main()
