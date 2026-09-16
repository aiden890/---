"""Diagnosis: run the TRAINING-loop rollout harness (grpo_train_loop._run_one_skill +
grpo_trainer_server op_sample, zero-init per-skill LoRA == base policy) for the GRASP
skill only, on the SAME seeds as the standalone 40% reference, sweeping BOTH the grasp
horizon and the sampler noise level eta. Answers task t_7b40aba0:

  * eta=0.0 (deterministic, == standalone sampler, sde_probe max_abs_diff 0.0): does the
    training harness reproduce the standalone GRASP success at the SAME horizon (208)?
    And does horizon_grasp=120 (training default) truncate them?  -> fair comparison.
  * eta>0 (e.g. 0.1, the value GRPO training actually uses for op_sample rollouts): what
    is the ACTUAL success rate the training loop sees per horizon?  -> "training reality".

Pure diagnosis: adapters are zero-init and NEVER updated (op=update is issued only with
EMPTY advantages, purely to clear the server's rollout store so CPU RAM stays bounded --
with no non-zero advantage no backward runs and opt.step() is skipped, so the resident
policy stays bit-for-bit the pretrained checkpoint). Reuses the train card's
_run_one_skill / TrainerClient verbatim -- no duplicated rollout logic.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/work")
sys.path.insert(0, "/skill_eval_tools")
sys.path.insert(0, "/rl_env/src")
sys.path.insert(0, "/train/src")

import rollout  # noqa: E402
import grpo_train_loop as gtl  # noqa: E402  (reuse the exact training rollout harness)
from reward import RewardConfig, RewardManager  # noqa: E402
from skill_manager import Skill, SkillOutcome  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trainer-port", type=int, default=10088)
    ap.add_argument("--server-addr", default="127.0.0.1")
    ap.add_argument("--model-path", default="/checkpoint")
    ap.add_argument("--split", default="pretrain",
                    help="scene split; standalone 40% reference used pretrain.")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--horizons", default="120,208",
                    help="grasp horizons to sweep (training default 120; standalone 208).")
    ap.add_argument("--etas", default="0.0,0.1",
                    help="sampler noise levels: 0.0 = deterministic (== standalone); "
                         "0.1 = the value GRPO training uses for its op_sample rollouts.")
    ap.add_argument("--save-video-horizon", type=int, default=208,
                    help="only save mp4s for this horizon to keep output small.")
    my, rest = ap.parse_known_args()
    args = rollout.parse_args(rest)
    rollout.validate_args(args)
    args.split = my.split
    args.horizon_move = 120
    args.horizon_place = 150
    args.max_skill_calls = 3
    seeds = [int(s) for s in my.seeds.split(",")]
    horizons = [int(h) for h in my.horizons.split(",")]
    etas = [float(e) for e in my.etas.split(",")]

    out = Path(my.out); out.mkdir(parents=True, exist_ok=True)
    reward_cfg = RewardConfig(mode="simulator", horizon=max(horizons), use_milestones=True)

    client = gtl.TrainerClient(my.model_path, my.server_addr, my.trainer_port,
                               args.robot_type, args.crop_ratio)

    report = {"task": "t_7b40aba0", "split": my.split,
              "policy": "pretrained base (zero-init LoRA, no update)",
              "harness": "grpo_train_loop._run_one_skill (training path)",
              "grasp_hold_steps": 20, "seeds": seeds, "horizons": horizons, "etas": etas,
              "results": {}}

    for eta in etas:
        etatag = f"eta{eta:g}"
        report["results"][etatag] = {}
        for H in horizons:
            args.horizon_grasp = H
            vid_dir = out / etatag / f"h{H}"; vid_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for seed in seeds:
                genv, sim = gtl._make_env(my.split, seed)
                try:
                    obs, _ = rollout.reset_env(genv, seed)
                    sim.rest_lid_pos = sim.lid_pos()
                    reward_mgr = RewardManager(reward_cfg)
                    save_vid = (H == my.save_video_horizon)
                    frames = [] if save_vid else None
                    vid = str(vid_dir / f"grasp_seed{seed}.mp4") if save_vid else None
                    # reproducible action-noise seed (same rule as eval_episode guard=1);
                    # at eta>0 the server derives a unique per-chunk noise from this.
                    eval_action_seed = int(seed) * 131 + 1
                    traj_id = f"diag_{etatag}_h{H}_s{seed}"
                    t0 = time.time()
                    fobs, outcome, r, steps, p, _hold = gtl._run_one_skill(
                        sim, client, obs, args, Skill.GRASP, reward_mgr, eta=eta,
                        traj_id=traj_id, seed=eval_action_seed,
                        frames=frames, save_video=vid)
                    dt = round(time.time() - t0, 1)
                    # at eta>0 the server stored rollout chunks; clear them with a benign
                    # empty-advantage update (no backward, no opt.step -> weights unchanged).
                    if eta > 0.0:
                        client.update({})
                    success = outcome is SkillOutcome.SUCCESS
                    fail = None if success else (
                        "never held lid lifted for 20 consecutive steps "
                        f"(contact={p.get('gripper_lid_contact')} lifted={p.get('lid_lifted')})")
                    row = {"seed": seed, "horizon": H, "eta": eta,
                           "success": bool(success),
                           "outcome": outcome.value if outcome else None,
                           "success_step": steps if success else None,
                           "steps": steps, "seconds": dt,
                           "final_contact": bool(p.get("gripper_lid_contact")),
                           "final_lifted": bool(p.get("lid_lifted")),
                           "final_grasped": bool(p.get("lid_grasped")),
                           "failure_reason": fail,
                           "video": (f"{etatag}/h{H}/grasp_seed{seed}.mp4" if save_vid else None)}
                    rows.append(row)
                    print(f"[{etatag} h{H}] seed={seed} outcome={row['outcome']} "
                          f"success_step={row['success_step']} steps={steps} ({dt}s)", flush=True)
                finally:
                    genv.close()
            n_succ = sum(rr["success"] for rr in rows)
            block = {"n": len(rows), "success_count": n_succ,
                     "success_rate": round(n_succ / len(rows), 3),
                     "success_seeds": [rr["seed"] for rr in rows if rr["success"]],
                     "rows": rows}
            report["results"][etatag][str(H)] = block
            (out / f"grasp_{etatag}_h{H}.json").write_text(json.dumps(block, indent=2))
            print(f"=== {etatag} horizon {H}: GRASP success {n_succ}/{len(rows)} "
                  f"({block['success_rate']*100:.0f}%) seeds={block['success_seeds']} ===",
                  flush=True)

    (out / "diag_summary.json").write_text(json.dumps(report, indent=2))
    client.close()
    print("=== DONE (grasp horizon x eta diag) ===", flush=True)


if __name__ == "__main__":
    main()
