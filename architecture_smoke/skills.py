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
        expected_instruction = contract.render_instruction(call.args)
        if call.instruction is not None and call.instruction != expected_instruction:
            raise ValueError(f"SkillCall({call.name}) instruction does not match registry")
        if call.contract is not None and call.contract != contract.done_description:
            raise ValueError(f"SkillCall({call.name}) contract does not match registry")
        if call.budget is not None:
            if isinstance(call.budget, bool) or not isinstance(call.budget, int) or call.budget <= 0:
                raise ValueError(f"SkillCall({call.name}) budget must be a positive integer")
            if call.budget > contract.max_steps:
                raise ValueError(
                    f"SkillCall({call.name}) budget {call.budget} exceeds {contract.max_steps}")
        return contract

    def typed_call(self, name: str, args=None, *, budget=None) -> SkillCall:
        """Build the canonical five-field call from this registry only."""
        contract = self.get(name)
        bound = dict(contract.default_args if args is None else args)
        contract.validate_args(bound)
        selected_budget = contract.max_steps if budget is None else budget
        call = SkillCall(
            name=name, args=bound,
            instruction=contract.render_instruction(bound),
            contract=contract.done_description, budget=selected_budget)
        self.validate_call(call)
        return call

    def render(self, call: SkillCall) -> str:
        contract = self.validate_call(call)
        return contract.render_instruction(call.args)

    def can_start(self, call: SkillCall, predicates: Mapping) -> bool:
        return self.get(call.name).can_start(predicates)
