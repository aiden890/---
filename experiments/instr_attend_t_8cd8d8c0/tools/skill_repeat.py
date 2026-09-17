"""Repeat selected skill rollouts from identical valid entry states."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

sys.path[:0] = ["/work", "/skilltools", "/tools"]
import rollout
import robocasa  # noqa: F401
from robocasa.utils.env_utils import convert_action
from skill_eval import Cond, Sim, hold_action, run_episode
from instr_probe import build_snapshots

INSTRUCTIONS = {
    "grasp": "Grasp the blender lid securely at its handle and lift it clear.",
    "place": "Place the blender lid on the blender, release it after it is stably supported, then move the gripper clear.",
}
HORIZONS = {"grasp": 208, "place": 96}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--skills", default="grasp,place")
    ap.add_argument("--server-addr", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=10086)
    ap.add_argument("--model-path", default="/checkpoint")
    ap.add_argument("--gen-attempts", type=int, default=6)
    ap.add_argument("--gen-horizon", type=int, default=900)
    my, rest = ap.parse_known_args()
    args = rollout.parse_args(rest)
    rollout.validate_args(args)
    args.seed, args.gen_attempts, args.gen_horizon = my.seed, my.gen_attempts, my.gen_horizon
    out = Path(my.out)
    out.mkdir(parents=True, exist_ok=True)
    skills = [x.strip() for x in my.skills.split(",") if x.strip()]

    client = rollout.EvalClient(my.model_path, my.server_addr, my.server_port, args.robot_type, args.crop_ratio)
    env = gym.make("robocasa/CloseBlenderLid", split=args.split, seed=my.seed)
    sim = Sim(env)
    try:
        obs, _ = rollout.reset_env(env, my.seed)
        sim.rest_lid_pos = sim.lid_pos()
        sim.rest_xy_to_closed = float(np.linalg.norm(sim.rest_lid_pos[:2] - sim.closed_pos()[:2]))
        snapshots = build_snapshots(client, sim, args, out) if "place" in skills else {}
        if "place" in skills and "place" not in snapshots:
            raise RuntimeError("could not create a valid PLACE entry state")

        conditions = {
            "grasp": Cond(lambda p: p["lid_grasped"], 20),
            "place": Cond(lambda p: p["official_check_success"], 1),
        }
        results = []
        for skill in skills:
            for run in range(1, my.repeats + 1):
                if skill == "grasp":
                    obs, _ = rollout.reset_env(env, my.seed)
                    sim.rest_lid_pos = sim.lid_pos()
                    start = "official reset(seed=9)"
                else:
                    sim.restore(snapshots["place"])
                    obs, _, _, _, _ = env.step(convert_action(hold_action(True)))
                    start = "shared valid pre-place snapshot"
                frames = [rollout.make_video_frame(obs)]
                stem = f"{skill}_run{run:02d}"
                result = run_episode(
                    sim, client, args, INSTRUCTIONS[skill], obs, HORIZONS[skill], conditions[skill],
                    out / f"{stem}.jsonl", frames,
                    {"kind": "repeat", "skill": skill, "run": run, "seed": my.seed},
                )
                imageio.mimsave(out / f"{stem}.mp4", frames, fps=args.video_fps)
                record = {"skill": skill, "run": run, "start_state": start,
                          "instruction": INSTRUCTIONS[skill], "video": f"{stem}.mp4", **result}
                (out / f"{stem}.json").write_text(json.dumps(record, indent=2, default=str))
                results.append(record)
                print(json.dumps({"skill": skill, "run": run, "success": result["success"],
                                  "steps": result["steps"], "reason": result["terminated_by"]}), flush=True)

        aggregate = {}
        for skill in skills:
            rows = [r for r in results if r["skill"] == skill]
            aggregate[skill] = {"runs": len(rows), "successes": sum(r["success"] for r in rows),
                                "success_rate": sum(r["success"] for r in rows) / len(rows)}
        (out / "summary.json").write_text(json.dumps({"seed": my.seed, "aggregate": aggregate,
                                                       "runs": results}, indent=2, default=str))
        print(json.dumps(aggregate), flush=True)
    finally:
        env.close()
        client.close()


if __name__ == "__main__":
    main()
