"""RLinf environment/reward adapter + MiBoT model registration.

Connects RoboCasa365 CloseBlenderLid + the verified simulator reward + skill entry-state
reset + GRASP/MOVE/PLACE skill contracts to RLinf's PPO/GRPO data flow, and registers the
MiBoT action model with the pinned RLinf commit.

Import-safe without a GPU or RLinf installed: RLinf and simulator objects are imported
lazily inside functions. The static shape/contract data is available for inspection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import mibot_adapter as MA

# Skill contracts (single source of truth mirrors the rl-env card FSM outcomes).
SKILL_CONTRACTS = {
    "GRASP_OBJECT": {
        "entry_state": "reset",                 # lid on counter
        "success": "lid_grasped held GRASP_HOLD_STEPS consecutive (contact AND lifted)",
        "no_angle_condition": True,
    },
    "MOVE_OBJECT": {
        "entry_state": "restored_post_grasp",   # already grasped + lifted
        "success": "N-step continuous in pre-place region",
    },
    "PLACE_OBJECT": {
        "entry_state": "restored_pre_place",    # already in pre-place region
        "success": "official_success (seated AND gripper-far>0.15m AND upright<=7deg), settle_terminal",
    },
}


@dataclass
class EnvSpec:
    task_name: str = "CloseBlenderLid"
    robots: str = "PandaOmron"
    skill: str = "GRASP_OBJECT"
    replan_steps: int = 16
    horizon: int = 400
    use_milestones: bool = False   # pay-once-at-end (user's chosen payout mode)


class RoboCasaRewardAdapter:
    """Wraps the verified RewardManager + SkillMonitor for RLinf step rewards.

    Reuses reward.py (definition) and skill_manager.py (FSM) by import — no reward logic
    is redefined here. RLinf calls step() with the per-step predicate dict produced by the
    simulator (skill_eval.Sim.predicates()).
    """

    def __init__(self, spec: EnvSpec):
        self.spec = spec
        self.reward_mgr = MA.load_reward_manager(use_milestones=spec.use_milestones)
        self.monitor = MA.load_skill_monitor(spec.skill)

    def reset(self):
        self.monitor = MA.load_skill_monitor(self.spec.skill)

    def step_reward(self, predicates: dict, step: int) -> dict:
        """Return {'reward': float, 'breakdown': ..., 'skill_outcome': ...} for one step."""
        rb = self.reward_mgr.step_reward(predicates, step)
        outcome = self.monitor.update(predicates, step)
        return {"reward": getattr(rb, "total", rb), "breakdown": rb, "skill_outcome": outcome}


def make_entry_state_reset(skill: str):
    """Return a reset callable that restores the skill's entry state via the verified
    skill_eval.Sim.snapshot()/restore() (qpos+qvel+gripper+blender flags).

    GRASP -> reset; MOVE -> restored post-grasp; PLACE -> restored pre-place.
    Requires the simulator (client image); NEEDS_SERVER for actual execution.
    """
    import skill_eval  # rollouts-xiaomi-t_4a072806/tools/skill_eval.py
    return skill_eval.make_entry_state_reset(skill) if hasattr(skill_eval, "make_entry_state_reset") else None


def register_mibot(rlinf_cfg: Optional[dict] = None) -> MA.RLinfModelSpec:
    """Register Xiaomi MiBoT as an RLinf embodied action model at the pinned commit.

    The MiBoT velocity field + pi-RL Flow-SDE sampler are exposed to RLinf's PPO/GRPO
    log-prob machinery via MiBoTSampler. This function performs the actual RLinf-side
    registration inside the container (import rlinf); NEEDS_SERVER to execute.

    Returns the RLinfModelSpec descriptor either way (inspectable statically).
    """
    spec = MA.RLinfModelSpec()
    try:
        import rlinf  # noqa: F401
        # Actual registration wiring lives against the pinned RLinf API. Kept as an
        # explicit NEEDS_SERVER seam rather than guessed code, so it fails loudly on the
        # server instead of silently drifting from the pinned commit's real API.
        raise NotImplementedError(
            "RLinf model registration is a NEEDS_SERVER gate: wire spec -> rlinf model "
            "registry against commit %s on the GPU server." % spec.framework_commit)
    except ImportError:
        pass
    return spec


__all__ = ["SKILL_CONTRACTS", "EnvSpec", "RoboCasaRewardAdapter",
           "make_entry_state_reset", "register_mibot"]
