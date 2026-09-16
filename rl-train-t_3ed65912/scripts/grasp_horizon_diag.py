"""Diagnosis: run the TRAINING-loop rollout harness (grpo_train_loop._run_one_skill +
grpo_trainer_server op_sample, eta=0, zero-init per-skill LoRA == base policy) for the
GRASP skill only, on the SAME seeds as the standalone 40% reference, sweeping the
grasp horizon. Answers task t_7b40aba0:

  * Does the training rollout harness reproduce the standalone GRASP success when given
    the SAME seeds + the SAME horizon (208)?  -> harness fidelity.
  * Does horizon_grasp=120 (the training default) truncate the standalone successes,
    whose 20-consecutive-hold GRASP completes at step 136/145/176/171?  -> horizon ablation.

This is pure diagnosis: eta=0, adapters zero-init (never updated), no op=update is ever
called, so the resident policy is bit-for-bit the pretrained checkpoint (sde_probe:
flow-SDE eta=0 == model.forward, max_abs_diff 0.0; LoRA B=0 => delta 0). Reuses the
train card's _run_one_skill / TrainerClient verbatim -- no duplicated rollout logic.
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
    ap.add_argument("--save-video-horizon", type=int, default=208,
                    help="only save mp4s for this horizon to keep output small.")
    my, rest = ap.parse_known_args()
    args = rollout.parse_args(rest)
    rollout.validate_args(args)
    # attributes _run_one_skill reads off args
    args.split = my.split
    args.horizon_move = 120
    args.horizon_place = 150
    args.max_skill_calls = 3
    args.obs_history = args.obs_history
    seeds = [int(s) for s in my.seeds.split(",")]
    horizons = [int(h) for h in my.horizons.split(",")]

    out = Path(my.out); out.mkdir(parents=True, exist_ok=True)
    reward_cfg = RewardConfig(mode="simulator", horizon=max(horizons), use_milestones=True)

    client = gtl.TrainerClient(my.model_path, my.server_addr, my.trainer_port,
                               args.robot_type, args.crop_ratio)

    report = {"task": "t_7b40aba0", "split": my.split, "eta": 0.0,
              "policy": "pretrained base (zero-init LoRA, no update)",
              "harness": "grpo_train_loop._run_one_skill (training path)",
              "grasp_hold_steps": 20, "seeds": seeds, "horizons": horizons,
              "results": {}}

    for H in horizons:
        args.horizon_grasp = H
        vid_dir = out / f"h{H}"; vid_dir.mkdir(parents=True, exist_ok=True)
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
                # deterministic paired action-noise seed, same rule as eval_episode (guard=1)
                eval_action_seed = int(seed) * 131 + 1
                t0 = time.time()
                fobs, outcome, r, steps, p = gtl._run_one_skill(
                    sim, client, obs, args, Skill.GRASP, reward_mgr, eta=0.0,
                    traj_id=None, seed=eval_action_seed,
                    frames=frames, save_video=vid)
                dt = round(time.time() - t0, 1)
                success = outcome is SkillOutcome.SUCCESS
                if success:
                    fail = None
                else:
                    fail = ("never held lid lifted for 20 consecutive steps "
                            f"(contact={p.get('gripper_lid_contact')} "
                            f"lifted={p.get('lid_lifted')})")
                row = {"seed": seed, "horizon": H,
                       "success": bool(success),
                       "outcome": outcome.value if outcome else None,
                       "success_step": steps if success else None,
                       "steps": steps, "seconds": dt,
                       "final_contact": bool(p.get("gripper_lid_contact")),
                       "final_lifted": bool(p.get("lid_lifted")),
                       "final_grasped": bool(p.get("lid_grasped")),
                       "failure_reason": fail,
                       "video": (f"h{H}/grasp_seed{seed}.mp4" if save_vid else None)}
                rows.append(row)
                print(f"[h{H}] seed={seed} outcome={row['outcome']} "
                      f"success_step={row['success_step']} steps={steps} ({dt}s)", flush=True)
            finally:
                genv.close()
        n_succ = sum(r["success"] for r in rows)
        report["results"][str(H)] = {
            "n": len(rows), "success_count": n_succ,
            "success_rate": round(n_succ / len(rows), 3),
            "success_seeds": [r["seed"] for r in rows if r["success"]],
            "rows": rows,
        }
        (out / f"grasp_h{H}.json").write_text(json.dumps(report["results"][str(H)], indent=2))
        print(f"=== horizon {H}: GRASP success {n_succ}/{len(rows)} "
              f"({report['results'][str(H)]['success_rate']*100:.0f}%) "
              f"seeds={report['results'][str(H)]['success_seeds']} ===", flush=True)

    (out / "diag_summary.json").write_text(json.dumps(report, indent=2))
    client.close()
    print("=== DONE (grasp horizon diag) ===", flush=True)


if __name__ == "__main__":
    main()
