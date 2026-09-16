"""G5 (success): faithful pi-RL Flow-SDE noise-level success sweep (sim rollout).

Operator decision A (2026-09-16): use the existing trainer now fitted with the
faithful pi-RL marginal-preserving Flow-SDE sampler (--sampler pirl) and run the
G5 noise-level success sweep -> G7 post-update ODE transfer. This is the SUCCESS
half of G5; the ODE-vs-SDE action-statistics half already passed in
rl-env/scripts/pirl_sampler_gpu_gates.py.

Purpose: for each pi-RL noise level (eta), measure the zero-shot per-skill success
rate over a fixed seed pool so we can pick the eta that BALANCES exploration
(eta>0 must still succeed sometimes, or GRPO gets no group-reward variance) against
COLLAPSE (too much noise -> policy never succeeds -> flat all-fail groups). eta=0 is
the deterministic ODE reference (== the pinned checkpoint's Euler flow, bit-exact).

Reuses grpo_train_loop by IMPORT (single source of truth for the rollout harness,
reward, skill FSM, env, TrainerClient). No forked rollout logic here.

Requires the trainer server up WITH --sampler pirl:
  bash run-train.sh trainer-start --sampler pirl --eta <ignored, set per-request> ...
Runs in the xiaomi-client image networked to the trainer (same as `train`).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# Reuse the training loop's imports + helpers as the single source of truth.
import grpo_train_loop as G
from grpo_train_loop import TrainerClient, _make_env, _run_one_skill, SKILL_KEY
from skill_manager import Skill, SkillOutcome
from reward import RewardConfig, RewardManager

SKILL_ENUM = {"grasp": Skill.GRASP, "move_holding": Skill.MOVE_HOLDING, "place": Skill.PLACE}


def _rollout_success(sim, client, args, skill, reward_cfg, seed, eta, action_seed):
    """One skill rollout at the given eta; return (success, outcome, reward, steps)."""
    reward_mgr = RewardManager(reward_cfg)
    _obs, outcome, r, steps, _p, _hold = _run_one_skill(
        sim, client, sim._reset_obs, args, skill, reward_mgr,
        eta=eta, traj_id=None, seed=action_seed)
    return bool(outcome is SkillOutcome.SUCCESS), (outcome.name if outcome else "NONE"), r, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trainer-port", type=int, default=10088)
    ap.add_argument("--server-addr", default="127.0.0.1")
    ap.add_argument("--model-path", default="/checkpoint")
    ap.add_argument("--split", default="target")
    ap.add_argument("--skill", default="grasp", choices=("grasp", "move_holding", "place"))
    ap.add_argument("--noise-sweep", default="0.0,0.1,0.3,0.5,0.7,1.0",
                    help="pi-RL noise levels (eta). 0.0 = deterministic ODE reference.")
    ap.add_argument("--n", type=int, default=12, help="seeds per noise level")
    ap.add_argument("--samples-per-seed", type=int, default=1,
                    help="stochastic rollouts per seed at eta>0 (eta=0 forced to 1)")
    ap.add_argument("--seed-base", type=int, default=5000)
    ap.add_argument("--reward-variant", default="simulator_terminal_only",
                    choices=("simulator_terminal_only", "simulator_milestones"))
    ap.add_argument("--horizon-grasp", type=int, default=120)
    ap.add_argument("--horizon-move", type=int, default=120)
    ap.add_argument("--horizon-place", type=int, default=150)
    my, rest = ap.parse_known_args()

    # Build the same `args` namespace the train loop uses (rollout defaults + our horizons).
    args = G.rollout.parse_args(rest)
    G.rollout.validate_args(args)
    for k in ("horizon_grasp", "horizon_move", "horizon_place", "split"):
        setattr(args, k, getattr(my, k))
    # attrs _run_one_skill / eval read but which live on `my` in the train loop
    args.save_videos = 0
    args.video_stride = getattr(args, "video_stride", 2)

    out = Path(my.out); out.mkdir(parents=True, exist_ok=True)
    skill = SKILL_ENUM[my.skill]
    reward_cfg = RewardConfig(mode="simulator", horizon=my.horizon_place,
                              use_milestones=(my.reward_variant == "simulator_milestones"))
    client = TrainerClient(my.model_path, my.server_addr, my.trainer_port,
                           args.robot_type, args.crop_ratio)
    etas = [float(x) for x in my.noise_sweep.split(",")]
    seeds = [my.seed_base + i for i in range(my.n)]

    report = {"config": {**vars(my), "skill": my.skill, "etas": etas, "seeds": seeds},
              "sweep": []}
    (out / "g5_progress.jsonl").write_text("")
    print(f"[G5] skill={my.skill} etas={etas} n={my.n} spp={my.samples_per_seed}", flush=True)

    with open(out / "g5_progress.jsonl", "a") as fh:
        for eta in etas:
            spp = 1 if eta == 0.0 else my.samples_per_seed
            n_succ = 0
            n_total = 0
            outcomes = {}
            returns = []
            t0 = time.time()
            for seed in seeds:
                for s in range(spp):
                    genv, sim = _make_env(my.split, seed)
                    try:
                        obs, _ = G.rollout.reset_env(genv, seed)
                        sim.rest_lid_pos = sim.lid_pos()
                        sim._reset_obs = obs
                        # deterministic per-(seed,sample) action-noise seed (paired across etas)
                        action_seed = int(seed) * 131 + 7 + s
                        ok, oc, r, steps = _rollout_success(
                            sim, client, args, skill, reward_cfg, seed, eta, action_seed)
                    finally:
                        genv.close()
                    n_total += 1
                    n_succ += int(ok)
                    outcomes[oc] = outcomes.get(oc, 0) + 1
                    returns.append(round(r, 4))
                    rec = {"eta": eta, "seed": seed, "sample": s, "success": ok,
                           "outcome": oc, "reward": round(r, 4), "steps": steps}
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    print(f"[G5] eta={eta} seed={seed}.{s} success={ok} outcome={oc} "
                          f"reward={r:.3f} steps={steps}", flush=True)
            rate = n_succ / n_total if n_total else 0.0
            row = {"eta": eta, "n_total": n_total, "n_success": n_succ,
                   "success_rate": round(rate, 4), "outcomes": outcomes,
                   "mean_return": round(float(np.mean(returns)), 4) if returns else 0.0,
                   "return_std": round(float(np.std(returns)), 4) if returns else 0.0,
                   "seconds": round(time.time() - t0, 1)}
            report["sweep"].append(row)
            print(f"[G5] === eta={eta}: success {n_succ}/{n_total}={rate:.3f} "
                  f"outcomes={outcomes} ({row['seconds']}s) ===", flush=True)
            (out / "g5_sweep.json").write_text(json.dumps(report, indent=2))

    # Pick recommended training eta: highest eta whose success_rate is in a healthy
    # exploration band (has signal but not collapsed). Report honestly if none qualifies.
    eta0 = next((r for r in report["sweep"] if r["eta"] == 0.0), None)
    base_rate = eta0["success_rate"] if eta0 else None
    cand = [r for r in report["sweep"]
            if r["eta"] > 0.0 and r["n_success"] > 0]
    rec = None
    if cand:
        # want highest exploration (largest eta) that still succeeds at least twice
        signal = [r for r in cand if r["n_success"] >= 2] or cand
        rec = max(signal, key=lambda r: (r["n_success"] >= 2, r["eta"]))
    report["base_eta0_rate"] = base_rate
    report["recommended_eta"] = rec["eta"] if rec else None
    report["recommendation_note"] = (
        f"eta0(ODE) success={base_rate}; recommended training eta={rec['eta'] if rec else None} "
        f"(success {rec['n_success']}/{rec['n_total']})" if rec else
        "No eta>0 produced any success on this seed pool -> no GRPO group-reward "
        "variance (all-fail groups). This skill is too hard from this entry-state for "
        "on-policy exploration at these noise levels; report honestly.")
    (out / "g5_sweep.json").write_text(json.dumps(report, indent=2))
    print("[G5] recommendation:", report["recommendation_note"], flush=True)
    client.close()
    print("=== DONE (g5-sweep) ===", flush=True)


if __name__ == "__main__":
    main()
