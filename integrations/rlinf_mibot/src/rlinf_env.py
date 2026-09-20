"""MiBoT model registration plus reusable skill-reward contracts for RLinf.

The native RLinf RoboCasa365 path currently uses RLinf's official full-task success reward.
The reward and entry-state helpers below expose the verified GRASP/MOVE/PLACE contracts for
the collector path and for a future native environment hook; merely constructing them does
not replace RLinf's native ``_calc_step_reward`` implementation.

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
    """Wrap the verified RewardManager + SkillMonitor for predicate-aware callers.

    Reuses reward.py (definition) and skill_manager.py (FSM) by import — no reward logic
    is redefined here. The async collector supplies ``skill_eval.Sim.predicates()``. The
    stock native RLinf environment does not yet expose that predicate dict to this adapter.
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

    The MiBoT velocity field + pi-RL Flow-SDE sampler are exposed to RLinf's GRPO
    log-prob machinery by :class:`mibot_rlinf_policy.MiBoTRLinfPolicy`.

    Returns the descriptor.  When RLinf is not installed this remains import-safe so
    host-side static checks can still inspect the integration.
    """
    spec = MA.RLinfModelSpec()
    try:
        from rlinf.models import register_model
    except ImportError:
        return spec

    # Do not swallow policy dependency/import failures once RLinf is present. A partially
    # configured GPU worker must fail at startup instead of appearing unregistered later.
    from mibot_rlinf_policy import build_mibot_policy

    # ``force`` makes registration idempotent in worker processes and Hydra relaunches.
    # RLinf also adds the dynamic name to SupportedModel/EMBODIED_MODEL.
    register_model(spec.name, build_mibot_policy, category="embodied", force=True)
    return spec


__all__ = ["SKILL_CONTRACTS", "EnvSpec", "RoboCasaRewardAdapter",
           "make_entry_state_reset", "register_mibot"]
