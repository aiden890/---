"""Sparse stable-grasp reward for the RLinf RoboCasa365 subprocess env."""
from __future__ import annotations

from typing import Any


class StableGraspTracker:
    """Latch success after ``hold_steps`` consecutive grasped environment steps."""

    def __init__(self, hold_steps: int = 20):
        if int(hold_steps) <= 0:
            raise ValueError("hold_steps must be positive")
        self.hold_steps = int(hold_steps)
        self.reset()

    def reset(self) -> None:
        self.streak = 0
        self.success = False

    def update(self, grasped: bool) -> bool:
        if self.success:
            return True
        self.streak = self.streak + 1 if bool(grasped) else 0
        if self.streak >= self.hold_steps:
            self.success = True
        return self.success


class StableGraspRewardEnv:
    """Delegate RoboCasa behavior while replacing task success with stable grasp.

    RLinf's RoboCasa subprocess worker calls ``_check_success`` on this outer object
    once after every delegated environment step. The returned boolean therefore becomes
    both the sparse reward event and termination. Success stays latched so the remainder
    of an already-issued action chunk cannot create a negative relative-reward transition.
    """

    def __init__(self, env: Any, hold_steps: int = 20, lift_dz: float = 0.05):
        self.env = env
        self.tracker = StableGraspTracker(hold_steps)
        self.lift_dz = float(lift_dz)
        self._rest_lid_z: float | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _lid(self):
        return self.env.blender.blender_lid

    def _lid_z(self) -> float:
        lid_body = f"{self._lid().name}_main"
        return float(self.env.sim.data.get_body_xpos(lid_body)[2])

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.tracker.reset()
        self._rest_lid_z = self._lid_z()
        return result

    def step(self, action):
        return self.env.step(action)

    def _is_grasped(self) -> bool:
        # Mirrors skill_eval.Sim.predicates(): gripper contact plus a lid lifted
        # strictly more than lift_dz from reset height and clear of the counter.
        from robocasa.models.fixtures import Counter

        robot = self.env.robots[0]
        gripper = robot.gripper["right"]
        lid = self._lid()
        contact = bool(self.env.check_contact(gripper, lid))
        on_counter = any(
            self.env.check_contact(lid, fixture)
            for fixture in self.env.fixtures.values()
            if isinstance(fixture, Counter)
        )
        lifted = (
            self._rest_lid_z is not None
            and self._lid_z() - self._rest_lid_z > self.lift_dz
            and not on_counter
        )
        return contact and lifted

    def _check_success(self) -> bool:
        return self.tracker.update(self._is_grasped())


def wrap_env_factory(env_fn, *, hold_steps: int = 20, lift_dz: float = 0.05):
    """Return a cloudpickle-friendly factory for RLinf's subprocess vector env."""

    def build():
        return StableGraspRewardEnv(
            env_fn(), hold_steps=hold_steps, lift_dz=lift_dz
        )

    return build


__all__ = ["StableGraspTracker", "StableGraspRewardEnv", "wrap_env_factory"]
