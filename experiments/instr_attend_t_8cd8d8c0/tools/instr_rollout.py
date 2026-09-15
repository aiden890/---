"""Visual rollout-divergence for the instruction-attendance probe (t_8cd8d8c0).

From ONE identical fixed start state (obs), run a short closed-loop rollout for each of a
few instructions and save per-instruction video + eef/lid trajectory. Same obs, different
instruction -> if trajectories diverge the policy attends; if they overlay it ignores.

Reuses /work/rollout.py + /skilltools/skill_eval.py (Sim, run_episode) by import.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, "/work")
sys.path.insert(0, "/skilltools")
import rollout  # noqa: E402
import skill_eval  # noqa: E402
from skill_eval import Sim, FULL_INSTRUCTION, SKILLS, hold_action, Cond  # noqa: E402
import gymnasium as gym  # noqa: E402
import robocasa  # noqa: E402,F401
from robocasa.utils.env_utils import convert_action  # noqa: E402


def f(x):
    return [float(v) for v in np.asarray(x).reshape(-1)]


NEVER = Cond(lambda p: False, 1)  # never "succeeds" -> always runs the full horizon


def rollout_instr(sim, client, args, instruction, obs, horizon, jsonl, mp4):
    frames = [rollout.make_video_frame(obs)]
    res = skill_eval.run_episode(sim, client, args, instruction, obs, horizon, NEVER, jsonl, frames,
                                 {"kind": "divergence", "instruction": instruction})
    imageio.mimsave(mp4, frames, fps=args.video_fps)
    eef = []
    for line in Path(jsonl).read_text().splitlines():
        r = json.loads(line)
        if r.get("type") == "step":
            eef.append(r["predicates"]["eef_pos"])
    return {"instruction": instruction, "steps": res["steps"], "video": Path(mp4).name,
            "eef_path": eef, "final_predicates": res["final_predicates"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--state", default="reset", choices=["reset", "move", "place"])
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--instructions", default="correct_full,skill_grasp,opposite_open_drawer,irrelevant_pick_cup")
    ap.add_argument("--gen-attempts", type=int, default=6)
    ap.add_argument("--gen-horizon", type=int, default=900)
    ap.add_argument("--server-addr", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=10086)
    ap.add_argument("--model-path", default="/checkpoint")
    my, rest = ap.parse_known_args()
    args = rollout.parse_args(rest)
    rollout.validate_args(args)
    args.seed = my.seed
    args.gen_attempts = my.gen_attempts
    args.gen_horizon = my.gen_horizon
    out = Path(my.out); out.mkdir(parents=True, exist_ok=True)

    battery = {"correct_full": FULL_INSTRUCTION, "skill_grasp": SKILLS["grasp"],
               "skill_move": SKILLS["move_holding"], "skill_place": SKILLS["place"],
               "opposite_open_drawer": "Open the drawer.", "irrelevant_pick_cup": "Pick up the cup.",
               "irrelevant_turn_on_stove": "Turn on the stove.", "empty": "", "nonsense": "asdf qwer zxcv lorem ipsum."}
    wanted = [s.strip() for s in my.instructions.split(",") if s.strip()]

    client = rollout.EvalClient(my.model_path, my.server_addr, my.server_port, args.robot_type, args.crop_ratio)
    genv = gym.make("robocasa/CloseBlenderLid", split=args.split, seed=my.seed)
    sim = Sim(genv)
    try:
        obs, _ = rollout.reset_env(genv, my.seed)
        sim.rest_lid_pos = sim.lid_pos()
        sim.rest_xy_to_closed = float(np.linalg.norm(sim.rest_lid_pos[:2] - sim.closed_pos()[:2]))

        snap = None
        if my.state != "reset":
            # reuse the probe's snapshot builder
            from instr_probe import build_snapshots
            snaps = build_snapshots(client, sim, args, out)
            if my.state not in snaps:
                raise RuntimeError(f"generation never produced {my.state} start state")
            snap = snaps[my.state]

        recs = {}
        for label in wanted:
            if my.state == "reset":
                obs, _ = rollout.reset_env(genv, my.seed)
            else:
                sim.restore(snap)
                obs, _, _, _, _ = genv.step(convert_action(hold_action(True)))
            recs[label] = rollout_instr(sim, client, args, battery[label], obs, my.horizon,
                                        out / f"div_{my.state}_{label}.jsonl", out / f"div_{my.state}_{label}.mp4")
            print(f"[div {my.state}] {label}: steps={recs[label]['steps']} "
                  f"eef_end={f(recs[label]['eef_path'][-1]) if recs[label]['eef_path'] else None}", flush=True)

        # pairwise eef-trajectory divergence vs correct_full
        def traj_l2(a, b):
            n = min(len(a), len(b))
            return float(np.mean([np.linalg.norm(np.array(a[i]) - np.array(b[i])) for i in range(n)])) if n else None
        ref = recs.get("correct_full", recs[wanted[0]])["eef_path"]
        div = {label: traj_l2(ref, recs[label]["eef_path"]) for label in wanted}
        (out / f"divergence_{my.state}.json").write_text(json.dumps(
            {"state": my.state, "seed": my.seed, "horizon": my.horizon,
             "mean_eef_traj_L2_vs_correct": div,
             "records": {k: {kk: recs[k][kk] for kk in ("instruction", "steps", "video")} for k in recs}},
            indent=2, default=str))
        print(f"[div {my.state}] mean eef-traj L2 vs correct_full: {json.dumps(div)}", flush=True)
    finally:
        genv.close(); client.close()


if __name__ == "__main__":
    main()
