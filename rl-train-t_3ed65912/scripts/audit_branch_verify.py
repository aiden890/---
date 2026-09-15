"""AUDIT items #4 + #10 integration test (GPU + simulator): real shared-prefix branching.

Runs inside the xiaomi-client image (RoboCasa sim + assets) against the running GRPO
trainer server (for actions). Verifies the two things the earlier NPZ-only "reproducibility"
test did NOT:

  #4a  FULL simulator-state restore round-trip. Roll an oracle-planner prefix, snapshot
       the sim with skill_eval.Sim.snapshot() (qpos+qvel+gripper action+blender flags+
       controller state -- the REAL restore path, not qpos-only NPZ), step forward, then
       restore() and require: (i) predicates identical to the snapshot, (ii) the SAME next
       transition for several steps when the SAME action sequence is replayed. This is the
       actual gate for the per-skill entry-state datasets (GRASP from reset, MOVE from a
       restored post-grasp state, PLACE from a restored pre-place state).

  #4b  SHARED-PREFIX GROUP BRANCHING. From one restored branch state, roll a group of G
       members with eta>0 (stochastic) and require they start from a byte-identical state
       (predicate + qpos equality at branch) yet diverge (distinct trajectories) -- the
       Z-1 shared-prefix property the GRPO group advantage assumes.

Writes results/audit/audit_branch_verify.json. Exit 0 iff both pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/work")
sys.path.insert(0, "/skill_eval_tools")
sys.path.insert(0, "/rl_env/src")
sys.path.insert(0, "/train/src")
import rollout  # noqa: E402
import skill_eval  # noqa: E402
import gymnasium as gym  # noqa: E402
import robocasa  # noqa: E402,F401
from robocasa.utils.env_utils import convert_action  # noqa: E402

from grpo_train_loop import TrainerClient, _make_env, _run_one_skill  # noqa: E402
from reward import RewardConfig, RewardManager  # noqa: E402
from skill_manager import (  # noqa: E402
    MonitorConfig, OraclePlanner, Skill, SkillMonitor, SkillOutcome, SKILL_INSTRUCTION,
)

SKILL_KEY = {Skill.GRASP: "grasp", Skill.MOVE_HOLDING: "move_holding", Skill.PLACE: "place"}


def predicates_equal(a, b, keys):
    return {k: (bool(a.get(k)) == bool(b.get(k))) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer-port", type=int, default=10088)
    ap.add_argument("--server-addr", default="127.0.0.1")
    ap.add_argument("--model-path", default="/checkpoint")
    ap.add_argument("--split", default="target")
    ap.add_argument("--seed", type=int, default=5004)   # a seed that grasps (from armB eval)
    ap.add_argument("--group", type=int, default=3)
    ap.add_argument("--eta", type=float, default=0.6)
    ap.add_argument("--replay-steps", type=int, default=8)
    ap.add_argument("--out", default="/out/audit_branch_verify.json")
    my, rest = ap.parse_known_args()
    args = rollout.parse_args(rest)
    rollout.validate_args(args)
    for k in ("group", "eta", "split"):
        setattr(args, k, getattr(my, k))
    # minimal fields _run_one_skill reads
    for k, v in {"horizon_grasp": 120, "horizon_move": 120, "horizon_place": 150,
                 "replan_steps": 16, "video_stride": 8, "video_fps": 10}.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    client = TrainerClient(my.model_path, my.server_addr, my.trainer_port,
                           args.robot_type, args.crop_ratio)
    rep = {"config": vars(my), "tests": {}}
    pred_keys = ["lid_grasped", "lid_lifted", "lid_on_blender", "in_preplace_region",
                 "gripper_lid_contact", "official_check_success"]

    # ---- roll an oracle GRASP prefix, then snapshot the real sim state ----
    genv, sim = _make_env(my.split, my.seed)
    obs, _ = rollout.reset_env(genv, my.seed)
    sim.rest_lid_pos = sim.lid_pos()
    reward_mgr = RewardManager(RewardConfig())
    obs, outcome, r, steps, p_prefix = _run_one_skill(
        sim, client, obs, args, Skill.GRASP, reward_mgr, eta=0.0, traj_id=None, seed=my.seed * 131 + 1)
    snap = sim.snapshot("branch_after_grasp", steps)
    snap_pred = dict(snap["predicates"])
    snap_qpos = snap["qpos"].copy()

    # ---- step forward a few actions to move AWAY from the snapshot ----
    replay_actions = []
    for i in range(my.replay_steps):
        a = skill_eval.hold_action(gripper_closed=True)
        replay_actions.append(a)
        obs, _, done, trunc, info = sim.genv.step(convert_action(a))
    moved_pred = sim.predicates()
    moved_qpos = sim.k.sim.get_state().qpos.copy()
    d_moved = float(np.abs(moved_qpos - snap_qpos).max())

    # ---- RESTORE and check predicate + qpos equality ----
    sim.restore(snap)
    rpred = sim.predicates()
    rqpos = sim.k.sim.get_state().qpos.copy()
    d_restore_qpos = float(np.abs(rqpos - snap_qpos).max())
    pred_match = predicates_equal(rpred, snap_pred, pred_keys)

    # ---- replay the SAME actions from the restored state; transitions must match ----
    qpos_traj_a = []
    for a in replay_actions:
        obs, _, done, trunc, info = sim.genv.step(convert_action(a))
        qpos_traj_a.append(sim.k.sim.get_state().qpos.copy())
    d_replay_vs_first = float(np.abs(qpos_traj_a[-1] - moved_qpos).max())

    rep["tests"]["fix4a_full_state_restore"] = {
        "snapshot_moved_away": d_moved > 1e-4, "d_moved": round(d_moved, 6),
        "restore_qpos_matches": d_restore_qpos <= 1e-6, "d_restore_qpos": d_restore_qpos,
        "restore_predicates_match": all(pred_match.values()), "pred_match": pred_match,
        "deterministic_replay_matches": d_replay_vs_first <= 1e-4,
        "d_replay_vs_first": round(d_replay_vs_first, 6),
    }
    f4a = rep["tests"]["fix4a_full_state_restore"]
    fix4a = (f4a["snapshot_moved_away"] and f4a["restore_qpos_matches"]
             and f4a["restore_predicates_match"] and f4a["deterministic_replay_matches"])
    genv.close()

    # ---- #4b: shared-prefix group branching from the SAME restored state ----
    branch_preds = []
    branch_qpos0 = []
    member_endpos = []
    for m in range(my.group):
        g2, s2 = _make_env(my.split, my.seed)
        try:
            o2, _ = rollout.reset_env(g2, my.seed)
            s2.rest_lid_pos = s2.lid_pos()
            rm = RewardManager(RewardConfig())
            # re-run the identical oracle GRASP prefix to reach the branch point
            o2, oc2, r2, st2, _ = _run_one_skill(
                s2, client, o2, args, Skill.GRASP, rm, eta=0.0, traj_id=None, seed=my.seed * 131 + 1)
            s2.restore(snap)  # identical branch state for every member
            branch_preds.append({k: bool(s2.predicates().get(k)) for k in pred_keys})
            branch_qpos0.append(s2.k.sim.get_state().qpos.copy())
            # stochastic MOVE_HOLDING rollout: members share the start, eta>0 diversifies
            o2, oc2, r2, st2, _ = _run_one_skill(
                s2, client, o2, args, Skill.MOVE_HOLDING, rm, eta=my.eta,
                traj_id=f"branch_m{m}", seed=1000 + m)
            member_endpos.append(s2.k.sim.get_state().qpos.copy())
        finally:
            g2.close()

    # all members must START identical...
    d_branch_start = max(float(np.abs(branch_qpos0[i] - branch_qpos0[0]).max())
                         for i in range(len(branch_qpos0)))
    preds_identical = all(branch_preds[i] == branch_preds[0] for i in range(len(branch_preds)))
    # ...and END distinct (stochastic divergence)
    end_pair = [float(np.abs(member_endpos[i] - member_endpos[0]).max()) for i in range(1, len(member_endpos))]
    d_branch_end = max(end_pair) if end_pair else 0.0

    rep["tests"]["fix4b_shared_prefix_branch"] = {
        "group": my.group,
        "members_start_identical": d_branch_start <= 1e-6 and preds_identical,
        "d_branch_start_qpos": d_branch_start, "start_predicates_identical": preds_identical,
        "members_diverge": d_branch_end > 1e-4, "d_branch_end_qpos": round(d_branch_end, 6),
    }
    f4b = rep["tests"]["fix4b_shared_prefix_branch"]
    fix4b = f4b["members_start_identical"] and f4b["members_diverge"]

    rep["fix4a_pass"] = bool(fix4a)
    rep["fix4b_pass"] = bool(fix4b)
    rep["overall_pass"] = bool(fix4a and fix4b)
    rep["scope"] = "audit_item4_simulator_branch_restore"
    out = Path(my.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))
    client.close()
    return 0 if rep["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
