"""Typed schemas for the adapter-free skill-conditioned inference architecture.

These dataclasses are the ONLY coupling between layers. planner, policy,
environment, verifier and executor each depend on these types, not on each
other's internals, so any single layer can be swapped for a mock or a real
implementation.

Pure stdlib -- importing this module never pulls in torch / gym / robocasa,
so the schema/render/handoff checks run locally without a GPU.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence


# --------------------------------------------------------------------------- #
#  Enums                                                                       #
# --------------------------------------------------------------------------- #
class SkillStatus(str, enum.Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


class Decision(str, enum.Enum):
    """Verifier handoff decision returned to the Execution Manager."""

    CONTINUE = "CONTINUE"   # keep executing the current skill window
    ADVANCE = "ADVANCE"     # done_when satisfied -> skill SUCCESS, hand back to planner
    RETRY = "RETRY"         # skill FAILED but recoverable -> planner may re-issue
    REPLAN = "REPLAN"       # skill TIMEOUT / stuck -> planner must re-plan


class AdapterMode(str, enum.Enum):
    DISABLED = "disabled"       # adapter-free smoke test: base policy only
    BASE_ONLY = "base_only"     # synonym accepted from the router
    LORA = "lora"               # NOT allowed in this smoke test


# --------------------------------------------------------------------------- #
#  Planner <-> Manager                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class PlannerContext:
    """Everything the planner is allowed to see. No sim / policy handles."""

    goal: str
    predicates: Mapping[str, Any]
    skill_catalog: Sequence[str]
    last_result: Optional["SkillResult"] = None
    step_budget_remaining: int = 0
    planner_calls: int = 0


@dataclass
class SkillCall:
    """Typed call the planner emits. It does NOT contain robot actions."""

    name: str
    args: Mapping[str, str]

    def as_dict(self) -> dict:
        return {"name": self.name, "args": dict(self.args)}


# --------------------------------------------------------------------------- #
#  Skill contract (definition -- no model inference)                          #
# --------------------------------------------------------------------------- #
@dataclass
class SkillContract:
    """A generic skill definition + its task binding.

    ``instruction_template`` is a ``str.format`` template rendered with the
    bound args. ``can_start`` / ``done_when`` are pure predicate functions over
    the environment predicate dict -- they contain the only task-specific logic
    and are kept out of the executor.
    """

    name: str
    description: str
    arg_schema: Sequence[str]                       # required arg keys
    instruction_template: str
    can_start: Callable[[Mapping[str, Any]], bool]
    done_when: Callable[[Mapping[str, Any]], bool]
    max_steps: int
    hold_steps: int = 1                             # consecutive done_when steps for stable_for(N)
    done_description: str = ""                       # human-readable done_when text
    default_args: Mapping[str, str] = field(default_factory=dict)

    def render_instruction(self, args: Mapping[str, str]) -> str:
        return self.instruction_template.format(**args)

    def validate_args(self, args: Mapping[str, str]) -> None:
        missing = [k for k in self.arg_schema if k not in args]
        if missing:
            raise ValueError(f"SkillCall({self.name}) missing args {missing}; got {dict(args)}")


# --------------------------------------------------------------------------- #
#  Manager <-> Policy                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class PolicyInput:
    """Inputs handed to the shared base VLA for one inference call."""

    instruction: str
    state_history: Any                              # np.ndarray, opaque here
    image_history: Mapping[str, Any]                # camera -> np.ndarray stack
    adapter_mode: AdapterMode = AdapterMode.DISABLED
    adapter_checkpoint: Optional[str] = None        # MUST be None in this smoke test


@dataclass
class PolicyOutput:
    action_chunk: Any                               # np.ndarray (>=replan, ACTION_DIM)
    chunk_len: int
    adapter_mode: AdapterMode
    adapter_checkpoint: Optional[str]


# --------------------------------------------------------------------------- #
#  Verifier -> Manager                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class VerificationResult:
    decision: Decision
    reason: str
    hold: int = 0
    elapsed: int = 0
    predicates: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Manager -> Planner                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class SkillResult:
    status: SkillStatus
    skill: str
    args: Mapping[str, str]
    instruction: str
    steps: int
    success_step: Optional[int]
    terminated_by: str
    reason: str
    predicates: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "status": self.status.value,
            "skill": self.skill,
            "args": dict(self.args),
            "instruction": self.instruction,
            "steps": self.steps,
            "success_step": self.success_step,
            "terminated_by": self.terminated_by,
            "reason": self.reason,
        }
