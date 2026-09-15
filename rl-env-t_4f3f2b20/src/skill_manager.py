"""Skill manager: the high-level plan + recovery FSM over the 3 parameterized skills.

High-level skill API (task t_4f3f2b20):
    GRASP(object)
    MOVE_HOLDING(object, destination, constraints)
    PLACE(object, destination, relation)

A "planner" emits a sequence of these calls; a skill monitor watches simulator
predicates to decide when a skill has succeeded, failed (and why), or timed out; and
the recovery policy maps a failure to the next skill call.

Two planner implementations share one interface (`Planner`):
  * OraclePlanner   -- deterministic predicate-driven planner used to VALIDATE the
                       environment end-to-end (this is the ground-truth "it works").
  * VLMPlannerStub  -- pluggable interface for a real VLM planner, evaluated
                       SEPARATELY from the oracle. Ships as a stub that mirrors the
                       oracle plan so the wiring is testable without a VLM; a real
                       planner subclasses it and overrides `propose`.

CloseBlenderLid canonical plan (task-specified):
    GRASP(blender_lid)
      -> MOVE_HOLDING(blender_lid, above_blender, collision_free)
      -> PLACE(blender_lid, blender, securely_on_top)

Recovery rules (task-specified):
    object_dropped                     -> restart from GRASP
    alignment lost during grasp/move   -> restart from MOVE_HOLDING
    released but lid off target        -> restart from GRASP
    lid_on_blender or gripper near      -> stay in PLACE (retreat continues)

The manager is simulator-agnostic: it consumes the predicate dict from
skill_eval.Sim.predicates() and the per-skill success/failure verdict, and returns
the next SkillCall. It does not step the env itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Skill(str, Enum):
    GRASP = "GRASP"
    MOVE_HOLDING = "MOVE_HOLDING"
    PLACE = "PLACE"


@dataclass
class SkillCall:
    skill: Skill
    obj: str
    destination: Optional[str] = None
    relation_or_constraints: Optional[str] = None

    def as_tuple(self):
        return (self.skill.value, self.obj, self.destination, self.relation_or_constraints)


# Natural-language instruction handed to the VLA executor per skill (matches the
# strings validated in t_4a072806/tools/skill_eval.py SKILLS).
SKILL_INSTRUCTION = {
    Skill.GRASP: "Pick up the blender lid securely.",
    Skill.MOVE_HOLDING: "Move the grasped blender lid above the blender without colliding with the surroundings.",
    Skill.PLACE: "Place the grasped blender lid securely on top of the blender, release it, and move the gripper away.",
}


class SkillOutcome(str, Enum):
    SUCCESS = "success"
    DROPPED = "object_dropped"
    ALIGNMENT_LOST = "alignment_lost"
    OFF_TARGET = "released_off_target"
    TIMEOUT = "timeout"
    COLLISION = "disallowed_collision"


# --------------------------------------------------------------------------- #
# Skill monitor: predicate -> success / failure-reason for a running skill.
# --------------------------------------------------------------------------- #
@dataclass
class MonitorConfig:
    grasp_hold_steps: int = 20
    move_hold_steps: int = 3


class SkillMonitor:
    """Watches predicates during one skill and produces a verdict."""

    def __init__(self, skill: Skill, config: MonitorConfig | None = None):
        self.skill = skill
        self.cfg = config or MonitorConfig()
        self._hold = 0
        self._was_grasped = False

    def update(self, p: dict[str, Any], step: int, horizon: int) -> Optional[SkillOutcome]:
        """Return a terminal SkillOutcome, or None to keep running."""
        grasped = bool(p.get("lid_grasped"))
        contact = bool(p.get("lid_grasped")) or bool(p.get("gripper_lid_contact"))

        if self.skill is Skill.GRASP:
            self._hold = self._hold + 1 if grasped else 0
            if self._hold >= self.cfg.grasp_hold_steps:
                return SkillOutcome.SUCCESS

        elif self.skill is Skill.MOVE_HOLDING:
            if self._was_grasped and not grasped and not p.get("lid_on_blender"):
                return SkillOutcome.DROPPED
            in_region = bool(p.get("in_preplace_region"))
            self._hold = self._hold + 1 if (grasped and in_region) else 0
            if self._hold >= self.cfg.move_hold_steps:
                return SkillOutcome.SUCCESS

        elif self.skill is Skill.PLACE:
            if bool(p.get("official_check_success")):
                return SkillOutcome.SUCCESS
            # lid slipped out of the hand before seating and is not on the blender
            if self._was_grasped and not grasped and not p.get("lid_on_blender") and not contact:
                return SkillOutcome.OFF_TARGET

        self._was_grasped = grasped
        if step >= horizon:
            return SkillOutcome.TIMEOUT
        return None


# --------------------------------------------------------------------------- #
# Planner interface + oracle + VLM stub.
# --------------------------------------------------------------------------- #
class Planner:
    """Base planner: proposes the next SkillCall from goal + current predicates + history."""

    name = "base"

    def initial(self) -> SkillCall:
        raise NotImplementedError

    def propose(self, last: SkillCall, outcome: SkillOutcome, p: dict[str, Any]) -> Optional[SkillCall]:
        """Return the next SkillCall, or None when the task is complete/aborted."""
        raise NotImplementedError


LID = "blender_lid"


class OraclePlanner(Planner):
    """Deterministic predicate-driven planner. Ground truth for end-to-end validation.

    Implements exactly the task's recovery rules. It never inspects images -- only the
    simulator predicate dict -- so a green end-to-end run with this planner proves the
    *environment* (executors, monitors, reward, recovery wiring) is correct,
    independently of any learned/VLM planner.
    """

    name = "oracle"

    PLAN = [
        SkillCall(Skill.GRASP, LID),
        SkillCall(Skill.MOVE_HOLDING, LID, "above_blender", "collision_free"),
        SkillCall(Skill.PLACE, LID, "blender", "securely_on_top"),
    ]

    def __init__(self, max_skill_calls: int = 12):
        self.max_skill_calls = max_skill_calls
        self._calls = 0

    def initial(self) -> SkillCall:
        self._calls = 1
        return self.PLAN[0]

    def _index_of(self, skill: Skill) -> int:
        for i, c in enumerate(self.PLAN):
            if c.skill is skill:
                return i
        raise KeyError(skill)

    def propose(self, last: SkillCall, outcome: SkillOutcome, p: dict[str, Any]) -> Optional[SkillCall]:
        self._calls += 1
        if self._calls > self.max_skill_calls:
            return None

        if outcome is SkillOutcome.SUCCESS:
            idx = self._index_of(last.skill)
            if idx + 1 < len(self.PLAN):
                return self.PLAN[idx + 1]
            return None  # PLACE succeeded -> task done

        # --- recovery rules (task-specified) ---
        if outcome is SkillOutcome.DROPPED:
            return self.PLAN[self._index_of(Skill.GRASP)]
        if outcome is SkillOutcome.ALIGNMENT_LOST:
            return self.PLAN[self._index_of(Skill.MOVE_HOLDING)]
        if outcome is SkillOutcome.OFF_TARGET:
            return self.PLAN[self._index_of(Skill.GRASP)]
        if outcome in (SkillOutcome.TIMEOUT, SkillOutcome.COLLISION):
            # If the lid is on/near the blender, keep retreating within PLACE.
            if p.get("lid_on_blender") or not p.get("gripper_lid_far_0.15", True):
                return self.PLAN[self._index_of(Skill.PLACE)]
            # otherwise, re-grasp
            return self.PLAN[self._index_of(Skill.GRASP)]
        return None


class VLMPlannerStub(Planner):
    """Interface for a real VLM planner, evaluated SEPARATELY from the oracle.

    The default implementation delegates to the oracle recovery logic so the pipeline
    is testable end-to-end without a VLM. A real planner overrides `propose` (and
    optionally `initial`) to call a vision-language model that reads the camera
    observation + goal and emits a structured SkillCall. The environment treats it as
    a black box and scores it with the same monitors/rewards, but never mixes its
    output into the oracle validation metric.
    """

    name = "vlm_stub"

    def __init__(self, backend: Any | None = None, max_skill_calls: int = 12):
        self.backend = backend          # a callable(goal, obs) -> SkillCall, or None
        self._oracle = OraclePlanner(max_skill_calls=max_skill_calls)

    def initial(self) -> SkillCall:
        if self.backend is not None:
            return self.backend("close_blender_lid", None)
        return self._oracle.initial()

    def propose(self, last: SkillCall, outcome: SkillOutcome, p: dict[str, Any]) -> Optional[SkillCall]:
        if self.backend is not None:
            return self.backend("close_blender_lid", {"last": last, "outcome": outcome, "predicates": p})
        return self._oracle.propose(last, outcome, p)


@dataclass
class SkillEpisodeResult:
    calls: list[dict] = field(default_factory=list)
    success: bool = False
    total_steps: int = 0
    planner: str = ""

    def add(self, call: SkillCall, outcome: SkillOutcome, steps: int, chunks: int):
        self.calls.append({
            "skill": call.skill.value,
            "obj": call.obj,
            "destination": call.destination,
            "relation_or_constraints": call.relation_or_constraints,
            "outcome": outcome.value,
            "steps": steps,
            "chunks": chunks,
        })
        self.total_steps += steps
