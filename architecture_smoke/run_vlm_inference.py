"""run_vlm_inference.py -- base-policy inference with an OBS-ONLY VLM verifier.

Task t_fc5e73d5. Runs the full skill-conditioned architecture for CloseBlenderLid
with the skill-termination judge REPLACED by the obs-only Qwen3-VL verifier:

    generic Qwen full-plan (once, pre-episode) -> SkillCall -> ExecutionManager
        -> base Xiaomi RoboCasa365 VLA (adapter DISABLED) -> RoboCasa env
        -> ObsVLMVerifier (policy's own frozen Qwen3-VL VQA, P(yes) latch)
        -> SkillResult -> planner,   for N randomized seeds.

The runtime judge reads ONLY what the robot receives -- 3 camera images + 14-D
proprio -- never a privileged simulator predicate. The sim ``official_check_success``
is read once at the end purely as the offline evaluation label. The VLM verifier
runs on a realistic cadence (proprio-event gate + a min step interval) and a
hysteresis latch, so it fires the moment the skill's goal is recognised.

Both the base-policy action forward AND the VLM VQA forward hit the SAME single
model load via ``infer_verify_server.py`` (policy-priority dispatcher), so the
5B backbone is loaded once. Run inside the xiaomi-client container with the RL
env card's ``vlm_scorer`` on the path (mounted at /rl_env/src).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from schemas import AdapterMode
import bindings
from skills import SkillRegistry
from pre_episode_planner import PreEpisodePlanner
from runtime_contract import POLICY_CHUNK_ACTIONS, as_dict as runtime_contract_dict
from environment import RoboCasaEnvironment
from executor import ExecutionManager
from trace import Trace
from episode import run_preplanned_episode


def build_registry() -> SkillRegistry:
    return SkillRegistry(bindings.CONTRACTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--split", default="pretrain")
    ap.add_argument("--server-addr", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=10086)
    ap.add_argument("--model-path", default="/checkpoint")
    ap.add_argument("--robot-type", default="robocasa365")
    ap.add_argument("--replan-steps", type=int, default=POLICY_CHUNK_ACTIONS)
    ap.add_argument("--obs-history", type=int, default=4)
    ap.add_argument("--obs-interval", type=int, default=2)
    ap.add_argument("--crop-ratio", type=float, default=0.95)
    ap.add_argument("--video-stride", type=int, default=2)
    ap.add_argument("--video-fps", type=int, default=20)
    ap.add_argument("--episode-budget", type=int, default=600)

    # obs-only VLM verifier knobs
    ap.add_argument("--vlm-min-interval", type=int, default=16,
                    help="min steps between VLM queries (realistic replan cadence)")
    ap.add_argument("--hysteresis-k", type=int, default=2,
                    help="consecutive confident-yes queries to latch ADVANCE")
    ap.add_argument("--tau", type=float, default=0.6, help="P(yes) threshold")
    ap.add_argument("--no-event-gate", action="store_true",
                    help="disable proprio candidate-stop gating (VLM cadence only)")
    ap.add_argument("--verifier-config", default=None,
                    help="JSON mapping skill names to per-skill view/tau/hysteresis_k/interval")
    ap.add_argument("--sync-verifier", action="store_true",
                    help="compatibility only: block the control loop on verifier RPCs")
    ap.add_argument("--planner-timeout", type=float, default=10.0)
    args = ap.parse_args()

    if args.verifier_config:
        config_payload = json.loads(Path(args.verifier_config).read_text())
        operating_points = config_payload.get("runtime_operating_points", config_payload)
    else:
        operating_points = {}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]

    from policy import BasePolicyClient
    from vlm_backends import RemoteVLMScorerBackend

    if args.replan_steps != POLICY_CHUNK_ACTIONS:
        ap.error(f"production runtime requires --replan-steps={POLICY_CHUNK_ACTIONS}")
    registry = build_registry()
    planner_kind = PreEpisodePlanner.kind
    policy = BasePolicyClient(
        model_path=args.model_path, server_addr=args.server_addr, server_port=args.server_port,
        robot_type=args.robot_type, crop_ratio=args.crop_ratio, replan_steps=args.replan_steps,
        adapter_mode=AdapterMode.DISABLED,
    )
    provenance = policy.provenance()
    assert provenance["adapter_checkpoint"] is None
    assert provenance["adapter_mode"] in ("disabled", "base_only")

    # obs-only VLM verifier backend: RPCs op=vlm_score to the SAME model load,
    # reusing the policy's processor (single source of truth for the VQA inputs).
    vlm_backend = RemoteVLMScorerBackend(
        policy.processor(), args.server_addr, args.server_port,
        robot_type=args.robot_type)

    config = {
        "task": "CloseBlenderLid", "goal": bindings.GOAL, "seeds": seeds, "split": args.split,
        "planner": {
            "mode": "pre_episode_full_plan", "kind": planner_kind,
            "calls_per_episode": 1, "boundary_calls": 0,
            "timeout_s": args.planner_timeout, "fallback": None,
            "json_repair": False, "sequence_substitution": False,
        },
        "verifier": {"kind": "obs_vlm", "backbone": "policy's own frozen Qwen3-VL VQA P(yes)",
                     "input": "3-cam images + 14D proprio ONLY (no sim predicate)",
                     "control_plane": ("synchronous_compatibility" if args.sync_verifier
                                       else "latest_only_async_default"),
                     "vlm_min_interval": args.vlm_min_interval, "hysteresis_k": args.hysteresis_k,
                     "tau": args.tau, "event_gated": not args.no_event_gate,
                     "per_skill_operating_points": operating_points},
        "adapter_mode": AdapterMode.DISABLED.value, "adapter_checkpoint": None,
        "policy_provenance": provenance,
        "replan_steps": args.replan_steps, "obs_history": args.obs_history,
        "obs_interval": args.obs_interval, "crop_ratio": args.crop_ratio,
        "video_stride": args.video_stride, "video_fps": args.video_fps,
        "episode_budget": args.episode_budget, "skill_catalog": registry.catalog(),
        "runtime_contract": runtime_contract_dict(),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))

    episodes = []
    try:
        for seed in seeds:
            seed_dir = out / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            env = RoboCasaEnvironment(
                seed=seed, split=args.split, replan_steps=args.replan_steps,
                obs_history=args.obs_history, obs_interval=args.obs_interval,
                crop_ratio=args.crop_ratio, video_stride=args.video_stride,
                video_fps=args.video_fps,
            )
            trace = Trace(seed_dir / "trace.jsonl", {
                "seed": seed, "task": "CloseBlenderLid", "adapter_mode": AdapterMode.DISABLED.value,
                "adapter_checkpoint": None, "planner_kind": planner_kind,
                "verifier_kind": "obs_vlm", "policy_provenance": provenance,
            })
            planner = PreEpisodePlanner(
                registry, vlm_backend, episode_id=f"seed{seed}",
                timeout_s=args.planner_timeout)
            try:
                env.reset()
                scene = env.scene_meta()
                (seed_dir / "scene.json").write_text(json.dumps(scene, indent=2, default=str))
                manager = ExecutionManager(
                    registry, policy, env, trace, AdapterMode.DISABLED,
                    vlm_backend=vlm_backend, vlm_min_interval=args.vlm_min_interval,
                    hysteresis_k=args.hysteresis_k, tau=args.tau,
                    event_gated=not args.no_event_gate,
                    synchronous_verifier=args.sync_verifier,
                    episode_id=f"seed{seed}",
                    verifier_operating_points=operating_points)
                summary = run_preplanned_episode(
                    planner, manager, env, trace, registry,
                    bindings.GOAL, args.episode_budget)
                nframes = env.save_video(seed_dir / "episode.mp4")
                summary["video"] = "episode.mp4"
                summary["video_frames"] = nframes
                (seed_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
                episodes.append(summary)
                print(json.dumps({
                    "seed": seed, "terminal": summary["terminal"],
                    "task_success": summary["task_success"],
                    "planner_calls": summary["planner_calls"],
                    "skills": [s["skill"] + ":" + s["status"] +
                               "(vlm@" + str((s.get("vlm_stats") or {}).get("succeeded_step")) + ")"
                               for s in summary["skills"]]}), flush=True)
            finally:
                trace.close()
                env.close()
    finally:
        try:
            vlm_backend.close()
        finally:
            policy.close()

    agg = {
        "task": "CloseBlenderLid", "verifier": "obs_vlm",
        "adapter_mode": AdapterMode.DISABLED.value, "adapter_checkpoint": None,
        "policy_provenance": provenance, "n_episodes": len(episodes), "seeds": seeds,
        "task_success_count": sum(1 for e in episodes if e["task_success"]),
        "episodes": episodes,
    }
    (out / "summary.json").write_text(json.dumps(agg, indent=2, default=str))
    print(json.dumps({"aggregate": {"n": len(episodes),
                                    "task_success": agg["task_success_count"]}}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
