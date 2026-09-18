"""Reward specification for CloseBlenderLid skill-conditioned RL.

Reward hierarchy (task t_4f3f2b20, strictly enforced here):

  PRIMARY  = simulator ground-truth predicates only. The official terminal success
             is  lid_on_blender == True  AND  gripper-lid distance > 0.15 m.
  SKILL    = per-skill predicate milestones (stable grasp, lid lifted, grasp
             maintained, collision-free transfer, pre-place reached, lid seated,
             released, retreated). Each milestone is paid AT MOST ONCE per episode
             (first time it becomes true), so the policy cannot farm a milestone by
             oscillating on its boundary.
  BASELINE = the Z-1-compatible signal: terminal 1/0 with success-aware temporal
             decay gamma = 0.998 (a late success is worth slightly less), kept so
             results are directly comparable to the Z-1 baseline runs.

The class returns a structured RewardBreakdown per step so every component is
auditable without a learned reward model.

Predicate names match rollouts-xiaomi-t_4a072806/tools/skill_eval.py Sim.predicates()
so this module consumes that exact dict without adaptation (single source of truth
for predicates -- we do not re-derive geometry here).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Official CloseBlenderLid success threshold constants (from robocasa source, mirrored
# in skill_eval.py): lid seated AND gripper far by >= 0.15 m AND lid upright.
GRIPPER_FAR_THRESH_M = 0.15


# Skill-milestone predicates: name -> function(predicate_dict) -> bool.
# These are the transition predicates the task enumerates.
SKILL_MILESTONES = {
    "stable_grasp": lambda p: bool(p.get("lid_grasped")),
    "lid_lifted": lambda p: bool(p.get("lid_lifted")),
    "grasp_maintained": lambda p: bool(p.get("lid_grasped")),  # tracked for duration in aggregator
    "collision_free_transfer": lambda p: bool(p.get("lid_grasped")) and not p.get("lid_other_contacts"),
    "pre_place_reached": lambda p: bool(p.get("in_preplace_region")),
    "lid_seated": lambda p: bool(p.get("lid_on_blender")),
    "released": lambda p: (not p.get("gripper_lid_contact")) and bool(p.get("lid_on_blender")),
    "retreated": lambda p: bool(p.get("gripper_lid_far_0.15")) and bool(p.get("lid_on_blender")),
}

# Default per-milestone bonus (dimensionless, small relative to terminal reward).
DEFAULT_MILESTONE_BONUS = {
    "stable_grasp": 0.15,
    "lid_lifted": 0.10,
    "collision_free_transfer": 0.10,
    "pre_place_reached": 0.20,
    "lid_seated": 0.25,
    "released": 0.10,
    "retreated": 0.10,
}


def official_success(p: dict[str, Any]) -> bool:
    """The one true terminal predicate. Prefers the env's own check when present."""
    if "official_check_success" in p:
        return bool(p["official_check_success"])
    return bool(p.get("lid_on_blender")) and bool(p.get("gripper_lid_far_0.15")) and bool(p.get("lid_upright_7deg", True))


@dataclass
class RewardConfig:
    terminal_success: float = 1.0
    persistent_success: bool = False         # if enabled, every frame in the official
                                            # success state receives 1 (not a one-frame pulse)
    success_stability_steps: int = 1         # consecutive closed-state env steps before success
    success_ignores_gripper_motion: bool = False
    terminal_decay_gamma: float = 0.998     # Z-1 success-aware decay
    horizon: int = 400                      # steps; used for the decay reference
    use_milestones: bool = False            # operator decision (2026-09-16): pay reward ONCE at the
                                            # end when ALL success conditions are met (terminal only),
                                            # instead of paying intermediate milestones piece by piece.
    settle_terminal: bool = True            # judge task success on the FINAL SETTLED state
                                            # (after the gripper has released and moved away),
                                            # not only during manipulation. A placement that
                                            # wobbles while releasing but comes to rest properly
                                            # closed (seated + upright + gripper away) still earns
                                            # the terminal reward. Evaluated at done/truncated.
    milestone_bonus: dict = field(default_factory=lambda: dict(DEFAULT_MILESTONE_BONUS))
    collision_requires_grasp: bool = True   # a collision is "disallowed" only while carrying
                                            # the lid; a resting lid touching its support
                                            # surface is NOT a collision (else a dense
                                            # per-step penalty saturates the RL reward and
                                            # kills GRPO advantage variance).
    penalties: dict = field(default_factory=lambda: {
        "object_dropped": -0.25,
        "disallowed_collision": -0.05,
        "timeout": -0.10,
    })

    @classmethod
    def binary(cls, *, horizon: int = 400) -> "RewardConfig":
        """Exact outcome reward: failed trajectory=0, successful trajectory=1."""
        return cls(
            terminal_success=1.0,
            persistent_success=True,
            success_stability_steps=10,      # 5 rendered frames at stride=2 (0.25 s at 20 fps)
            success_ignores_gripper_motion=True,
            settle_terminal=False,
            terminal_decay_gamma=1.0,
            horizon=horizon,
            use_milestones=False,
            penalties={
                "object_dropped": 0.0,
                "disallowed_collision": 0.0,
                "timeout": 0.0,
            },
        )


@dataclass
class RewardBreakdown:
    step: int
    terminal: float = 0.0
    milestone: float = 0.0
    penalty: float = 0.0
    object_dropped: float = 0.0
    disallowed_collision: float = 0.0
    timeout: float = 0.0
    milestones_fired: list[str] = field(default_factory=list)
    success: bool = False

    @property
    def primary(self) -> float:
        """Total simulator reward (terminal + milestone + penalty)."""
        return self.terminal + self.milestone + self.penalty


class RewardManager:
    """Stateful per-episode reward computer. Enforces once-per-episode milestones."""

    def __init__(self, config: RewardConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._fired: set[str] = set()
        self._terminal_paid = False
        self._success_streak = 0
        self._success_latched = False
        self._prev_grasped = False

    def step_reward(
        self,
        step_index: int,
        predicates: dict[str, Any],
        *,
        done: bool = False,
        truncated: bool = False,
    ) -> RewardBreakdown:
        p = predicates
        rb = RewardBreakdown(step=step_index)

        # --- milestones (once each) ---
        if self.cfg.use_milestones:
            for name, fn in SKILL_MILESTONES.items():
                if name == "grasp_maintained":
                    continue  # duration-based, handled by the aggregator, not paid here
                if name not in self._fired and fn(p):
                    self._fired.add(name)
                    bonus = self.cfg.milestone_bonus.get(name, 0.0)
                    rb.milestone += bonus
                    if bonus:
                        rb.milestones_fired.append(name)

        # --- penalties ---
        # object dropped: was grasped, now neither grasped nor seated nor in contact.
        dropped = self._prev_grasped and not p.get("lid_grasped") and not p.get("lid_on_blender") \
            and not p.get("gripper_lid_contact")
        if dropped:
            rb.object_dropped = self.cfg.penalties["object_dropped"]
            rb.penalty += rb.object_dropped
        if p.get("lid_other_contacts") and (not self.cfg.collision_requires_grasp or p.get("lid_grasped")):
            rb.disallowed_collision = self.cfg.penalties["disallowed_collision"]
            rb.penalty += rb.disallowed_collision
        self._prev_grasped = bool(p.get("lid_grasped"))

        # --- terminal success (paid once) with Z-1 success-aware decay ---
        # settle_terminal: the task is judged on the SETTLED state. official_success already
        # requires the gripper to be far (>0.15 m), so it can only be true once the hand has
        # released and moved away -- i.e. this is exactly "properly closed after manipulation
        # is finished". A placement that wobbles while releasing but comes to rest seated +
        # upright still becomes True here and is paid; one that ends tilted or off-position
        # never does. At episode end (done/truncated) we RE-EVALUATE on the final predicates
        # so a success that only stabilises on the very last step is not missed.
        if self.cfg.success_ignores_gripper_motion:
            # CloseBlenderLid success is the stable closed/upright lid state. Gripper or
            # robot motion is deliberately irrelevant. Debounce it, then latch for the
            # rest of the episode so the displayed current reward remains 1.
            closed = bool(p.get("lid_on_blender")) and bool(p.get("lid_upright_7deg", True))
            self._success_streak = self._success_streak + 1 if closed else 0
            if self._success_streak >= max(1, int(self.cfg.success_stability_steps)):
                self._success_latched = True
            success = self._success_latched
        else:
            success = official_success(p)
        at_end = bool(truncated or done)
        pay_now = success and (self.cfg.persistent_success or not self._terminal_paid)
        if self.cfg.settle_terminal:
            # only pay while the hand is clear of the lid (settled), or at episode end
            pay_now = pay_now and (bool(p.get("gripper_lid_far_0.15")) or at_end)
        rb.success = success
        if pay_now:
            self._terminal_paid = True
            decay = 1.0 if self.cfg.persistent_success else self.cfg.terminal_decay_gamma ** step_index
            rb.terminal += self.cfg.terminal_success * decay

        # timeout penalty at the end of a failed episode
        if at_end and not self._terminal_paid:
            rb.timeout = self.cfg.penalties["timeout"]
            rb.penalty += rb.timeout

        return rb


def z1_baseline_reward(success: bool, steps: int, gamma: float = 0.998) -> float:
    """The pure Z-1-compatible terminal reward: 1/0 with success-aware decay."""
    return (gamma ** steps) if success else 0.0


# ============================================================================
# Boundary-compliance ("post-success hold") reward  --  operator decision B (run30)
# ----------------------------------------------------------------------------
# The confirmed failure mode is NOT "the policy cannot do the skill" but "after the
# skill's success predicate is met it does NOT STOP -- it drifts into the next action"
# (hold experiment: KEEPS_MOVING; skill-boundary not respected). SFT (behaviour cloning)
# does not fix this; the operator's directive is to teach boundary respect directly with
# a closed-loop RL reward: once a skill has succeeded, REWARD staying put and PENALISE
# further end-effector motion.
#
# This is the reward DEFINITION (reusable, unit-tested). It is applied only AFTER the
# SkillMonitor first reports SUCCESS, over a post-success hold window, using the same
# eef position the predicates already expose ("eef_pos") -- single source of truth, no
# new geometry. It is a TRAINING-ONLY shaping term (eval uses the pure sim reward), just
# like approach shaping / timeout penalty.
# ============================================================================

@dataclass
class HoldConfig:
    """Per-step boundary-compliance shaping applied during the post-success hold window."""
    stay_bonus: float = 0.05          # + per step while the eef stays within stay_radius_m
    drift_penalty: float = 0.10       # - scaled by how far past stay_radius_m the eef moved
    stay_radius_m: float = 0.02       # <= this per-step eef displacement counts as "stopped"
    drop_success_penalty: float = 0.5  # - if the success predicate LAPSES during the hold
                                       # (drifting broke the achieved state -> boundary violation)


def hold_step_reward(eef_prev, eef_curr, still_success: bool, cfg: HoldConfig) -> float:
    """One post-success-hold step's boundary reward.

    eef_prev, eef_curr : 3-vectors (any sequence of 3 floats) = eef position last/this step.
    still_success      : is the skill's success predicate STILL true this step?
    Returns:
      + stay_bonus                         when displacement <= stay_radius_m (held position)
      - drift_penalty * (disp/radius - 1)  when displacement >  stay_radius_m (drifted away)
      - drop_success_penalty (additional)  when the success state has lapsed (still_success False)
    So standing still after success is rewarded and drifting/breaking the state is penalised,
    which is exactly the boundary-respect signal SFT could not instill.
    """
    import math
    dx = float(eef_curr[0]) - float(eef_prev[0])
    dy = float(eef_curr[1]) - float(eef_prev[1])
    dz = float(eef_curr[2]) - float(eef_prev[2])
    disp = math.sqrt(dx * dx + dy * dy + dz * dz)
    if disp <= cfg.stay_radius_m:
        r = cfg.stay_bonus
    else:
        r = -cfg.drift_penalty * (disp / max(cfg.stay_radius_m, 1e-6) - 1.0)
    if not still_success:
        r -= cfg.drop_success_penalty
    return float(r)
