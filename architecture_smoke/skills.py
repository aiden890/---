"""Generic skill registry -- contract lookup, arg validation, instruction render.

No model inference, no simulator. The registry is populated with task bindings
(bindings.CONTRACTS for CloseBlenderLid) but the class itself is task-agnostic.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from schemas import SkillCall, SkillContract


class SkillRegistry:
    def __init__(self, contracts: Sequence[SkillContract]):
        self._by_name = {c.name: c for c in contracts}
        if len(self._by_name) != len(contracts):
            raise ValueError("duplicate skill names in contract list")

    @property
    def names(self) -> list[str]:
        return list(self._by_name)

    def get(self, name: str) -> SkillContract:
        if name not in self._by_name:
            raise KeyError(f"unknown skill {name!r}; catalog={self.names}")
        return self._by_name[name]

    def catalog(self) -> list[dict]:
        out = []
        for c in self._by_name.values():
            out.append({
                "name": c.name,
                "description": c.description,
                "args": list(c.arg_schema),
                "instruction": c.instruction_template,
                "done_when": c.done_description,
                "max_steps": c.max_steps,
            })
        return out

    def validate_call(self, call: SkillCall) -> SkillContract:
        contract = self.get(call.name)
        contract.validate_args(call.args)
        return contract

    def render(self, call: SkillCall) -> str:
        contract = self.validate_call(call)
        return contract.render_instruction(call.args)

    def can_start(self, call: SkillCall, predicates: Mapping) -> bool:
        return self.get(call.name).can_start(predicates)
