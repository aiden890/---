"""Goal-2: compose the policy conditioning for each skill / ablation arm.

The Xiaomi checkpoint's only text conditioning slot is the `instruction` string
passed to `client.infer(states, images, instruction)` (deploy/client.py builds the
VLM prompt `... Generate robot actions for the task:\n{instruction} /no_cot`). This
module turns a (skill_id, arm) into the instruction text (+ a skill_id for the
per-skill LoRA / learned embedding). It is the single source of truth for how the
overall task goal + stable skill_id + NL instruction + object/destination/constraints
map onto the policy, shared by the SFT dataloader, the eval driver, and the RL loop.
"""
from __future__ import annotations

import json
import os
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCHEMA_PATH = os.path.join(_HERE, "..", "configs", "conditioning_schema.json")

ARMS = ("nl_only", "nl_plus_skill_id", "shared_lora_learned_embedding")


def load_schema(path: str = _SCHEMA_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def skill_id(schema: dict, skill: str) -> int:
    return schema["skills"][skill]["skill_id"]


def compose_structured_instruction(schema: dict, skill: str) -> str:
    s = schema["skills"][skill]
    fmt = schema["structured_instruction_format"]
    return fmt.format(
        overall_task_goal=schema["overall_task_goal"],
        skill_id=s["skill_id"], skill_name=skill,
        instruction_natural_language=s["instruction_natural_language"],
        object=s["object"], destination=s["destination"],
        constraints="; ".join(s["constraints"]),
    )


def build_instruction(schema: dict, skill: str, arm: str) -> str:
    """Return the instruction text for a (skill, ablation arm)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    if arm == "nl_only":
        return schema["skills"][skill]["instruction_natural_language"]
    if arm == "nl_plus_skill_id":
        return compose_structured_instruction(schema, skill)
    # shared_lora_learned_embedding: text held to the overall goal; the skill signal
    # enters through the learned embedding keyed by skill_id, not the text.
    return schema["overall_task_goal"]


def conditioning(schema: dict, skill: str, arm: str) -> dict:
    """Full conditioning bundle for a (skill, arm): text + skill_id + embedding flag."""
    return {
        "skill": skill,
        "skill_id": skill_id(schema, skill),
        "arm": arm,
        "instruction": build_instruction(schema, skill, arm),
        "use_learned_embedding": arm == "shared_lora_learned_embedding",
        "lora_key": "shared" if arm == "shared_lora_learned_embedding" else skill,
    }


if __name__ == "__main__":
    sc = load_schema()
    for arm in ARMS:
        print("=" * 60, arm)
        for sk in sc["skills"]:
            c = conditioning(sc, sk, arm)
            print(f"[{c['skill_id']}] {sk} (lora={c['lora_key']}, emb={c['use_learned_embedding']}):")
            print("   ", repr(c["instruction"]))
