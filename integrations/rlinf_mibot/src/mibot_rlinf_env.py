"""Merge-friendly RLinf RoboCasa365 extensions for MiBoT temporal observations.

The collector branch already has the production snapshot/group scheduler. This module
deliberately does not duplicate it. It adds only the policy-facing temporal history that
RLinf's stock RoboCasa365 environment currently lacks, and keeps the same observation keys
so the native MiBoT policy can consume either current frames or real history.

No vendored RLinf source is modified. ``install_mibot_env_override`` replaces the class
lookup only when ``env.*.mibot_history.enabled`` is true.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Callable, Iterable


def observation_to_mibot_state(
    observation: dict[str, Any], key_map: dict[str, str] | None = None
):
    """Build the checkpoint's exact 14-D EE-first state from a raw RoboCasa obs.

    RLinf's stock RoboCasa365 state concatenates raw quaternions, gripper velocity and
    other fields.  MiBoT was trained on base-relative EEF position/axis-angle, gripper
    qpos, and base position/axis-angle.  Delegate the quaternion conversion and ordering
    to the already-verified deployment transform instead of maintaining a second copy.
    """
    import rollout

    keys = key_map or {}

    def value(component: str, default: str):
        return observation[keys.get(component, default)]

    return rollout.observation_to_state(
        {
            "state.end_effector_position_relative": value(
                "base_to_eef_pos", "robot0_base_to_eef_pos"
            ),
            "state.end_effector_rotation_relative": value(
                "base_to_eef_quat", "robot0_base_to_eef_quat"
            ),
            "state.gripper_qpos": value("gripper_qpos", "robot0_gripper_qpos"),
            "state.base_position": value("base_pos", "robot0_base_pos"),
            "state.base_rotation": value("base_quat", "robot0_base_quat"),
        }
    )


def prepare_mibot_env_actions(actions: Any, *, binarize: bool = True):
    """Match RoboCasa's Xiaomi gym-wrapper action conversion for flat 12-D actions.

    RLinf currently applies this conversion only to its built-in OpenPI model names.
    Applying it again is harmless for already-binarized values, which also makes this
    adapter safe if MiBoT is added to RLinf's built-in family in a later revision.
    """
    if not binarize:
        return actions
    try:
        import torch

        if torch.is_tensor(actions):
            converted = actions.clone()
            if converted.shape[-1] >= 12:
                converted[..., 6] = torch.where(
                    converted[..., 6] < 0.5, -1.0, 1.0
                )
                converted[..., 11] = torch.where(
                    converted[..., 11] < 0.5, -1.0, 1.0
                )
            return converted
    except ImportError:
        pass

    import numpy as np

    converted = np.asarray(actions).copy()
    if converted.shape[-1] >= 12:
        converted[..., 6] = np.where(converted[..., 6] < 0.5, -1.0, 1.0)
        converted[..., 11] = np.where(converted[..., 11] < 0.5, -1.0, 1.0)
    return converted


class TemporalHistoryBuffer:
    """Per-environment fixed-stride history with reset-safe padding.

    The internal queue length is ``(length - 1) * interval + 1``. A reset fills that
    entire queue with the first observation, matching the legacy rollout behavior.
    Values remain generic so the reset/stride contract is unit-testable without torch.
    """

    def __init__(self, num_envs: int, length: int = 4, interval: int = 2):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if length <= 0 or interval <= 0:
            raise ValueError("history length and interval must be positive")
        self.num_envs = int(num_envs)
        self.length = int(length)
        self.interval = int(interval)
        self.queue_length = (self.length - 1) * self.interval + 1
        self._queues: dict[str, list[deque[Any]]] = {}

    def _key_queues(self, key: str) -> list[deque[Any]]:
        if key not in self._queues:
            self._queues[key] = [
                deque(maxlen=self.queue_length) for _ in range(self.num_envs)
            ]
        return self._queues[key]

    def initialized(self, key: str, env_id: int) -> bool:
        return bool(self._key_queues(key)[int(env_id)])

    def reset(self, key: str, env_id: int, value: Any) -> None:
        queue = self._key_queues(key)[int(env_id)]
        queue.clear()
        queue.extend([value] * self.queue_length)

    def append(self, key: str, env_id: int, value: Any) -> None:
        queue = self._key_queues(key)[int(env_id)]
        if not queue:
            self.reset(key, env_id, value)
        else:
            queue.append(value)

    def values(self, key: str, env_id: int) -> list[Any]:
        queue = self._key_queues(key)[int(env_id)]
        if not queue:
            raise RuntimeError(f"history {key!r} for env {env_id} is uninitialized")
        items = list(queue)
        indices = [
            len(items) - 1 - offset * self.interval
            for offset in reversed(range(self.length))
        ]
        return [items[index] for index in indices]

    def batch(self, key: str, stack: Callable[[list[Any]], Any]) -> Any:
        return stack(
            [stack(self.values(key, env_id)) for env_id in range(self.num_envs)]
        )


_MIBOT_ENV_CLASS = None


def build_mibot_robocasa365_env_class():
    """Build the subclass lazily so this module remains import-safe off the GPU host."""
    global _MIBOT_ENV_CLASS
    if _MIBOT_ENV_CLASS is not None:
        return _MIBOT_ENV_CLASS

    import torch
    from rlinf.envs.sim.robocasa365.robocasa365_env import Robocasa365Env

    class MiBoTHistoryRobocasa365Env(Robocasa365Env):
        """Stock RoboCasa365 environment with reset-aware strided tensor history."""

        def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
            history_cfg = cfg.get("mibot_history", {}) or {}
            self._mibot_history = TemporalHistoryBuffer(
                num_envs=num_envs,
                length=int(history_cfg.get("length", 4)),
                interval=int(history_cfg.get("interval", 2)),
            )
            self._mibot_reset_ids: set[int] | None = None
            super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

        def get_env_fns(self, env_idx=None):
            env_fns = super().get_env_fns(env_idx=env_idx)
            reward_cfg = self.cfg.get("grasp_binary_reward", {}) or {}
            if not bool(reward_cfg.get("enabled", False)):
                return env_fns
            from grasp_binary_reward import wrap_env_factory

            return [
                wrap_env_factory(
                    env_fn,
                    hold_steps=int(reward_cfg.get("hold_steps", 20)),
                    lift_dz=float(reward_cfg.get("lift_dz", 0.05)),
                )
                for env_fn in env_fns
            ]

        def _extract_state_vector(self, obs_single):
            key_map = self.observation_cfg.get("state_key_map", {}) or {}
            return observation_to_mibot_state(obs_single, key_map)

        def step(self, actions, auto_reset=True):
            actions = prepare_mibot_env_actions(
                actions,
                binarize=bool(
                    self.action_space_cfg.get("binarize_gripper_control", True)
                ),
            )
            return super().step(actions, auto_reset=auto_reset)

        def reset(self, env_idx=None, options=None):
            if env_idx is None:
                ids = set(range(self.num_envs))
            elif isinstance(env_idx, int):
                ids = {int(env_idx)}
            else:
                ids = {int(index) for index in env_idx}
            self._mibot_reset_ids = ids
            try:
                return super().reset(env_idx=env_idx, options=options)
            finally:
                self._mibot_reset_ids = None

        def _wrap_obs(self, obs_list, info_list):
            obs = super()._wrap_obs(obs_list, info_list)
            update_ids: Iterable[int]
            if self._mibot_reset_ids is None:
                update_ids = range(self.num_envs)
                reset_ids: set[int] = set()
            else:
                update_ids = sorted(self._mibot_reset_ids)
                reset_ids = self._mibot_reset_ids

            for key in ("main_images", "wrist_images", "extra_view_images", "states"):
                batch_value = obs.get(key)
                if batch_value is None:
                    continue
                for env_id in update_ids:
                    value = batch_value[env_id].detach().clone()
                    if env_id in reset_ids or not self._mibot_history.initialized(
                        key, env_id
                    ):
                        self._mibot_history.reset(key, env_id, value)
                    else:
                        self._mibot_history.append(key, env_id, value)
                obs[key] = self._mibot_history.batch(key, lambda xs: torch.stack(xs, dim=0))
            return obs

    _MIBOT_ENV_CLASS = MiBoTHistoryRobocasa365Env
    return _MIBOT_ENV_CLASS


def install_mibot_env_override() -> None:
    """Route opted-in ``robocasa365`` configs to the history-preserving subclass."""
    import rlinf.envs as envs
    import rlinf.workers.env.env_worker as env_worker

    if getattr(envs.get_env_cls, "_mibot_history_override", False):
        return
    original_get_env_cls = envs.get_env_cls

    def get_env_cls(env_type: str, env_cfg=None):
        history_cfg = env_cfg.get("mibot_history", {}) if env_cfg is not None else {}
        if str(env_type) == "robocasa365" and bool(history_cfg.get("enabled", False)):
            return build_mibot_robocasa365_env_class()
        return original_get_env_cls(env_type, env_cfg)

    get_env_cls._mibot_history_override = True
    get_env_cls._mibot_original = original_get_env_cls
    envs.get_env_cls = get_env_cls
    # EnvWorker imports the function by name, so patch its bound reference as well.
    env_worker.get_env_cls = get_env_cls


__all__ = [
    "TemporalHistoryBuffer",
    "observation_to_mibot_state",
    "prepare_mibot_env_actions",
    "build_mibot_robocasa365_env_class",
    "install_mibot_env_override",
]
