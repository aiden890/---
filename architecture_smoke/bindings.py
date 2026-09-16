"""CloseBlenderLid task bindings -- the ONLY task-specific data.

Kept separate from generic orchestration (schemas/executor/verifier) and from
the generic skill machinery (skills.py). Predicate names here must match the
keys produced by the environment adapter's ``predicates()`` (which forwards the
parent ``skill_eval.Sim.predicates()`` dict verbatim -- single source of truth).

Thresholds mirror the verified skill_eval.py constants:
  LIFT_DZ=0.05, PREPLACE_XY=0.06, GRASP_HOLD_STEPS=20, MOVE_HOLD_STEPS=3.
"""
from __future__ import annotations

from typing import Any, Mapping

from schemas import SkillContract

GOAL = "Close the blender lid securely."
FULL_TASK_INSTRUCTION = "Close the lid blender by securely placing the lid on top."

# Exact defined instructions from the RL design page (tracking/inference.json).
GRASP_INSTRUCTION = "Grasp the blender lid securely at its handle and lift it clear."
MOVE_INSTRUCTION = (
    "Move the held blender lid to the pre-place region above the blender while "
    "maintaining a stable grasp and avoiding disallowed contact."
)
PLACE_INSTRUCTION = (
    "Place the blender lid on the blender, release it after it is stably supported, "
    "then move the gripper clear."
)


# ---- predicate helpers (over the env predicate dict) ---------------------- #
def _grasped(p: Mapping[str, Any]) -> bool:
    return bool(p.get("lid_grasped"))


def _in_preplace(p: Mapping[str, Any]) -> bool:
    return bool(p.get("in_preplace_region"))


def _official_success(p: Mapping[str, Any]) -> bool:
    return bool(p.get("official_check_success"))


# ---- CloseBlenderLid skill contracts -------------------------------------- #
# Generic skill names GRASP_OBJECT / MOVE_OBJECT / PLACE_OBJECT are bound to the
# blender-lid objects here; the templates render a fixed natural-language string.
CONTRACTS = [
    SkillContract(
        name="GRASP_OBJECT",
        description="Grasp an object at a named grasp region and lift it clear.",
        arg_schema=("object", "grasp_region"),
        instruction_template=GRASP_INSTRUCTION,
        can_start=lambda p: not _grasped(p),
        done_when=lambda p: _grasped(p),
        max_steps=208,
        hold_steps=20,   # stable_for(20)
        done_description="held AND lifted>=0.05m AND stable_for(20) AND no_disallowed_contact",
        default_args={"object": "blender_lid", "grasp_region": "lid_handle"},
    ),
    SkillContract(
        name="MOVE_OBJECT",
        description="Move a held object to a named destination region.",
        arg_schema=("object", "destination"),
        instruction_template=MOVE_INSTRUCTION,
        can_start=lambda p: _grasped(p),
        done_when=lambda p: _grasped(p) and _in_preplace(p),
        max_steps=288,
        hold_steps=3,    # stable_for(3)
        done_description="held AND at_target(xy<=0.06m,z 0.015-0.25m) AND stable_for(3) AND no_disallowed_contact",
        default_args={"object": "blender_lid", "destination": "closed_preplace_region"},
    ),
    SkillContract(
        name="PLACE_OBJECT",
        description="Place a held object on a support, release, and retreat.",
        arg_schema=("object", "destination"),
        instruction_template=PLACE_INSTRUCTION,
        can_start=lambda p: _grasped(p),
        done_when=lambda p: _official_success(p),
        max_steps=96,
        hold_steps=1,
        done_description="supported_by AND released AND upright_error<=7deg AND gripper_clear>=0.15m",
        default_args={"object": "blender_lid", "destination": "blender"},
    ),
]
