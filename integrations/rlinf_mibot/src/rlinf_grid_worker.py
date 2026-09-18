"""RLinf-native RoboCasa rollout worker for the MiBoT parameter-grid collector."""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
import time
from pathlib import Path

# Canonical implementations are mounted read-only by integrations/rlinf_mibot/run.sh.
for candidate in ("/train/src", "/work", "/skill_eval_tools", "/rl_env/src"):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from rlinf.scheduler import Worker
from grid_collection import jsonable


class RoboCasaGridWorker(Worker):
    """Thin RLinf worker: simulator lifecycle only; inference stays in one model server."""

    def __init__(self, runtime_cfg: dict):
        super().__init__()
        self.runtime_cfg = dict(runtime_cfg)
        self.client = None

    def _client(self):
        if self.client is None:
            from grpo_train_loop import TrainerClient
            server = self.runtime_cfg["model_server"]
            self.client = TrainerClient(
                server.get("model_path", "/checkpoint"),
                server.get("host", "127.0.0.1"),
                int(server.get("port", 10088)),
                server.get("robot_type", "robocasa365"),
                float(server.get("crop_ratio", 0.95)),
            )
        return self.client

    @staticmethod
    def _args(parameters: dict) -> argparse.Namespace:
        horizon = int(parameters.get("env.horizon", 250))
        return argparse.Namespace(
            split=parameters.get("env.split", "pretrain"),
            horizon_grasp=int(parameters.get("env.horizon_grasp", horizon)),
            horizon_move=int(parameters.get("env.horizon_move", horizon)),
            horizon_place=int(parameters.get("env.horizon_place", horizon)),
            obs_history=int(parameters.get("rollout.obs_history", 4)),
            obs_interval=int(parameters.get("rollout.obs_interval", 2)),
            replan_steps=int(parameters.get("rollout.replan_steps", 16)),
            video_stride=int(parameters.get("rollout.video_stride", 2)),
            video_fps=int(parameters.get("rollout.video_fps", 20)),
            reward_variant=parameters.get("reward.variant", "simulator_terminal_only"),
            skill_success_reward=float(parameters.get("reward.skill_success", 1.0)),
            skill_success_gamma=float(parameters.get("reward.success_gamma", 0.998)),
            skill_success_decay=bool(parameters.get("reward.success_decay", True)),
        )

    def run_episode(self, job: dict) -> dict:
        from grpo_train_loop import SKILL_KEY, _make_env, _run_one_skill
        from reward import RewardConfig, RewardManager
        from skill_manager import Skill, SkillOutcome
        import rollout

        started = time.time()
        parameters = job["parameters"]
        injection_seed = self.runtime_cfg.get("validation", {}).get("inject_fail_once_seed")
        if injection_seed is not None and int(job["seed"]) == int(injection_seed):
            marker = (Path(self.runtime_cfg["results_root"]) / self.runtime_cfg["run_id"] /
                      f".injected-failure-seed-{job['seed']}")
            try:
                fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                pass
            else:
                os.close(fd)
                raise RuntimeError(f"validation-injected worker failure for seed {job['seed']}")
        args = self._args(parameters)
        skill = {
            "GRASP_OBJECT": Skill.GRASP,
            "MOVE_OBJECT": Skill.MOVE_HOLDING,
            "MOVE_HOLDING": Skill.MOVE_HOLDING,
            "PLACE_OBJECT": Skill.PLACE,
        }[parameters.get("env.skill", "GRASP_OBJECT")]
        eta = float(parameters.get("sampler.noise_level", 0.3))
        trainer = self._client()
        trainer_memory_before = trainer.metrics()
        attempt = int(job.get("_attempt", 0)) + 1
        traj_id = f"{job['job_id']}__attempt{attempt}"
        genv, sim = _make_env(args.split, int(job["seed"]))
        episode_dir = (Path(self.runtime_cfg["results_root"]) / self.runtime_cfg["run_id"] /
                       job["config_id"] / "episodes" / job["job_id"] /
                       f"attempt-{attempt}")
        episode_dir.mkdir(parents=True, exist_ok=True)
        try:
            obs, _ = rollout.reset_env(genv, int(job["seed"]))
            sim.rest_lid_pos = sim.lid_pos()
            snapshot_blob = pickle.dumps(sim.snapshot(), protocol=pickle.HIGHEST_PROTOCOL)
            randomization = {
                "env_seed": int(job["seed"]),
                "split": args.split,
                "initial_lid_pos": jsonable(sim.lid_pos()),
                "initial_predicates": jsonable(sim.predicates()),
                "initial_snapshot_sha256": hashlib.sha256(snapshot_blob).hexdigest(),
            }
            reward_cfg = RewardConfig(
                horizon=max(args.horizon_grasp, args.horizon_move, args.horizon_place),
                use_milestones=bool(parameters.get("reward.use_milestones", False)),
            )
            reward_mgr = RewardManager(reward_cfg)
            frames = []
            video_path = episode_dir / "rollout.mp4"
            _, outcome, reward, steps, predicates, hold = _run_one_skill(
                sim, trainer, obs, args, skill, reward_mgr,
                eta=eta, traj_id=traj_id, seed=int(job["seed"]),
                approach_coef=0.0, timeout_penalty=0.0, hold_cfg=None, hold_steps=0,
                frames=frames, save_video=str(video_path),
            )
            payload_path = (Path(self.runtime_cfg["results_root"]) / self.runtime_cfg["run_id"] /
                            job["config_id"] / "payloads" /
                            f"{job['job_id']}__attempt{attempt}.pt")
            trainer_payload = trainer.export_store([traj_id], payload_path, drop_after_export=True)
            trainer_payload["optimizer_update_requested"] = False
            trainer_memory_after = trainer.metrics()
            return {
                "started_at": started,
                "finished_at": time.time(),
                "worker_id": f"rlinf-{self._rank}",
                "attempt": attempt,
                "steps": int(steps),
                "skill_outcome": outcome.name if outcome else "NONE",
                "success": bool(outcome is SkillOutcome.SUCCESS),
                "reward": float(reward),
                "randomization": randomization,
                "final_predicates": jsonable(predicates),
                "hold_stats": jsonable(hold),
                "timing": {"wall_seconds": time.time() - started},
                "trainer_memory_before": trainer_memory_before,
                "trainer_memory_after": trainer_memory_after,
                "trainer_payload": trainer_payload,
                "artifacts": {"video": str(video_path),
                              "trainer_store": str(payload_path)},
                "skill_key": SKILL_KEY[skill],
            }
        except Exception:
            try:
                trainer.discard_store([traj_id])
            except Exception:
                pass
            raise
        finally:
            genv.close()

    def health(self):
        metrics = self._client().metrics()
        return {"worker_id": f"rlinf-{self._rank}", "model_server": metrics}

    def close_client(self):
        if self.client is not None:
            self.client.close()
            self.client = None
        return True
