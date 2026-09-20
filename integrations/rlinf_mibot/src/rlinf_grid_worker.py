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
        self._env = None
        self._sim = None
        self._env_split = None

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

    def _trajectory_id(self, actor_id: str, job: dict, attempt: int) -> str:
        # Job IDs repeat across independently collected rollout waves. Include the
        # immutable run ID so an adaptive multi-wave batch cannot overwrite an older
        # trajectory in the learner store.
        run_id = str(self.runtime_cfg["run_id"])
        return f"{actor_id}/{run_id}/{job['job_id']}__attempt{attempt}"

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
                return {
                    "worker_error": f"RuntimeError: validation-injected worker failure for seed {job['seed']}",
                    "started_at": started,
                    "finished_at": time.time(),
                }
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
        episode_policy_version = int(trainer_memory_before["policy_version"])
        actor_id = str(self.runtime_cfg.get("actor_id", f"rlinf-{self._rank}"))
        attempt = int(job.get("_attempt", 0)) + 1
        traj_id = self._trajectory_id(actor_id, job, attempt)
        reuse_env = bool(self.runtime_cfg.get("runtime", {}).get("reuse_env", True))
        if not reuse_env or self._env is None or self._env_split != args.split:
            if self._env is not None:
                self._env.close()
            self._env, self._sim = _make_env(args.split, int(job["seed"]))
            self._env_split = args.split
        genv, sim = self._env, self._sim
        episode_dir = (Path(self.runtime_cfg["results_root"]) / self.runtime_cfg["run_id"] /
                       job["config_id"] / "episodes" / job["job_id"] /
                       f"attempt-{attempt}")
        episode_dir.mkdir(parents=True, exist_ok=True)
        try:
            obs, _ = rollout.reset_env(genv, int(job["seed"]))
            sim.rest_lid_pos = sim.lid_pos()
            snapshot_blob = pickle.dumps(
                sim.snapshot("grid_initial", 0), protocol=pickle.HIGHEST_PROTOCOL)
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
            save_rollout_video = bool(parameters.get("rollout.save_video", False))
            frames = [] if save_rollout_video else None
            video_path = episode_dir / "rollout.mp4"
            _, outcome, reward, steps, predicates, hold = _run_one_skill(
                sim, trainer, obs, args, skill, reward_mgr,
                eta=eta, traj_id=traj_id, seed=int(job["seed"]),
                approach_coef=0.0, timeout_penalty=0.0, hold_cfg=None, hold_steps=0,
                frames=frames, save_video=str(video_path) if save_rollout_video else None,
                expected_policy_version=episode_policy_version,
            )
            payload_path = (Path(self.runtime_cfg["results_root"]) / self.runtime_cfg["run_id"] /
                            job["config_id"] / "payloads" /
                            f"{job['job_id']}__attempt{attempt}.pt")
            trainer_payload = trainer.export_store(
                [traj_id], payload_path, drop_after_export=True, actor_id=actor_id,
                group_id=str(job.get("group_id", job["config_id"])))
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
                "reward": (float(parameters.get("reward.skill_success", 1.0))
                           if outcome is SkillOutcome.SUCCESS and
                           parameters.get("reward.variant", "simulator_terminal_only") ==
                           "simulator_terminal_only" and float(reward) == 0.0 else float(reward)),
                "randomization": randomization,
                "final_predicates": jsonable(predicates),
                "hold_stats": jsonable(hold),
                "timing": {"wall_seconds": time.time() - started},
                "trainer_memory_before": trainer_memory_before,
                "trainer_memory_after": trainer_memory_after,
                "trainer_payload": trainer_payload,
                "artifacts": {"video": str(video_path) if save_rollout_video else None,
                              "trainer_store": str(payload_path)},
                "skill_key": SKILL_KEY[skill],
            }
        except Exception as exc:
            try:
                trainer.discard_store([traj_id])
            except Exception:
                pass
            if reuse_env and self._env is not None:
                self._env.close()
                self._env = self._sim = self._env_split = None
            # RLinf treats an actor exception as process-fatal. Return a serializable failure
            # envelope so the collector can apply its bounded retry policy instead.
            return {
                "worker_error": f"{type(exc).__name__}: {exc}",
                "started_at": started,
                "finished_at": time.time(),
            }
        finally:
            if not reuse_env:
                genv.close()
                self._env = self._sim = self._env_split = None

    def run_group(self, jobs: list[dict]) -> dict:
        """Run one GRPO group from one reset/snapshot on one persistent worker."""
        from grpo_train_loop import SKILL_KEY, _current_obs, _make_env, _run_one_skill
        from reward import RewardConfig, RewardManager
        from skill_manager import Skill, SkillOutcome
        import rollout

        group_started = time.time()
        if not jobs:
            return {"worker_error": "ValueError: empty rollout group", "results": []}
        first = jobs[0]
        identity = (first.get("group_id"), first["config_id"], int(first["seed"]),
                    first["parameters"])
        if any((job.get("group_id"), job["config_id"], int(job["seed"]), job["parameters"])
               != identity for job in jobs):
            return {"worker_error": "ValueError: mixed jobs in rollout group", "results": []}

        parameters = first["parameters"]
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
        policy_version = int(trainer_memory_before["policy_version"])
        actor_id = str(self.runtime_cfg.get("actor_id", f"rlinf-{self._rank}"))
        reuse_env = bool(self.runtime_cfg.get("runtime", {}).get("reuse_env", True))
        if not reuse_env or self._env is None or self._env_split != args.split:
            if self._env is not None:
                self._env.close()
            self._env, self._sim = _make_env(args.split, int(first["seed"]))
            self._env_split = args.split
        genv, sim = self._env, self._sim
        trajectory_ids = []
        try:
            rollout.reset_env(genv, int(first["seed"]))
            sim.rest_lid_pos = sim.lid_pos()
            initial_snapshot = sim.snapshot("grid_initial", 0)
            snapshot_blob = pickle.dumps(initial_snapshot, protocol=pickle.HIGHEST_PROTOCOL)
            randomization = {
                "env_seed": int(first["seed"]),
                "split": args.split,
                "initial_lid_pos": jsonable(sim.lid_pos()),
                "initial_predicates": jsonable(sim.predicates()),
                "initial_snapshot_sha256": hashlib.sha256(snapshot_blob).hexdigest(),
            }
            results = []
            for job in sorted(jobs, key=lambda item: int(item["member_index"])):
                member_started = time.time()
                attempt = int(job.get("_attempt", 0)) + 1
                traj_id = self._trajectory_id(actor_id, job, attempt)
                trajectory_ids.append(traj_id)
                sim.restore(initial_snapshot)
                obs = _current_obs(sim)
                episode_dir = (Path(self.runtime_cfg["results_root"]) /
                               self.runtime_cfg["run_id"] / job["config_id"] / "episodes" /
                               job["job_id"] / f"attempt-{attempt}")
                episode_dir.mkdir(parents=True, exist_ok=True)
                reward_cfg = RewardConfig(
                    horizon=max(args.horizon_grasp, args.horizon_move, args.horizon_place),
                    use_milestones=bool(parameters.get("reward.use_milestones", False)),
                )
                save_video = bool(parameters.get("rollout.save_video", False))
                frames = [] if save_video else None
                video_path = episode_dir / "rollout.mp4"
                _, outcome, reward, steps, predicates, hold = _run_one_skill(
                    sim, trainer, obs, args, skill, RewardManager(reward_cfg),
                    eta=eta, traj_id=traj_id,
                    seed=int(job.get("action_seed", job["seed"])),
                    approach_coef=0.0, timeout_penalty=0.0, hold_cfg=None, hold_steps=0,
                    frames=frames, save_video=str(video_path) if save_video else None,
                    expected_policy_version=policy_version,
                )
                payload_path = (Path(self.runtime_cfg["results_root"]) /
                                self.runtime_cfg["run_id"] / job["config_id"] / "payloads" /
                                f"{job['job_id']}__attempt{attempt}.pt")
                trainer_payload = trainer.export_store(
                    [traj_id], payload_path, drop_after_export=True, actor_id=actor_id,
                    group_id=str(job["group_id"]))
                trainer_payload["optimizer_update_requested"] = False
                results.append({
                    "job_id": job["job_id"],
                    "started_at": member_started,
                    "finished_at": time.time(),
                    "worker_id": f"rlinf-{self._rank}",
                    "attempt": attempt,
                    "steps": int(steps),
                    "skill_outcome": outcome.name if outcome else "NONE",
                    "success": bool(outcome is SkillOutcome.SUCCESS),
                    "reward": (float(parameters.get("reward.skill_success", 1.0))
                               if outcome is SkillOutcome.SUCCESS and
                               parameters.get("reward.variant", "simulator_terminal_only") ==
                               "simulator_terminal_only" and float(reward) == 0.0
                               else float(reward)),
                    "randomization": randomization,
                    "final_predicates": jsonable(predicates),
                    "hold_stats": jsonable(hold),
                    "timing": {"wall_seconds": time.time() - member_started,
                               "group_wall_seconds": time.time() - group_started},
                    "trainer_memory_before": trainer_memory_before,
                    "trainer_memory_after": trainer.metrics(),
                    "trainer_payload": trainer_payload,
                    "artifacts": {"video": str(video_path) if save_video else None,
                                  "trainer_store": str(payload_path)},
                    "skill_key": SKILL_KEY[skill],
                })
            return {"results": results, "group_id": first["group_id"],
                    "worker_id": f"rlinf-{self._rank}",
                    "group_wall_seconds": time.time() - group_started}
        except Exception as exc:
            try:
                trainer.discard_store(trajectory_ids)
            except Exception:
                pass
            if reuse_env and self._env is not None:
                self._env.close()
                self._env = self._sim = self._env_split = None
            return {"worker_error": f"{type(exc).__name__}: {exc}", "results": [],
                    "started_at": group_started, "finished_at": time.time()}
        finally:
            if not reuse_env:
                genv.close()
                self._env = self._sim = self._env_split = None

    def health(self):
        metrics = self._client().metrics()
        return {"worker_id": f"rlinf-{self._rank}", "model_server": metrics}

    def close_client(self):
        if self._env is not None:
            self._env.close()
            self._env = self._sim = self._env_split = None
        if self.client is not None:
            self.client.close()
            self.client = None
        return True
