"""Predicate verifier -- owns skill termination / handoff.

Evaluates a SkillContract's done_when (with its hold_steps = stable_for(N)) and
the step budget, and returns ADVANCE / CONTINUE / RETRY / REPLAN. It contains
NO model inference and does not select the next skill (that is the planner's
job); it only judges the current skill window and hands control back.

The verifier is stateful only across one skill execution (it counts the
consecutive done_when hold), matching skill_eval.py's ``hold`` semantics.
"""
from __future__ import annotations

from typing import Mapping

from schemas import Decision, SkillContract, VerificationResult


class PredicateVerifier:
    def __init__(self, contract: SkillContract):
        self.contract = contract
        self.hold = 0
        self.elapsed = 0
        self.succeeded_step: int | None = None

    def check_can_start(self, predicates: Mapping) -> bool:
        return self.contract.can_start(predicates)

    def update(self, predicates: Mapping) -> VerificationResult:
        """Call once per executed environment step."""
        self.elapsed += 1
        done_now = self.contract.done_when(predicates)
        self.hold = self.hold + 1 if done_now else 0
        latched = self.hold >= self.contract.hold_steps

        if latched:
            if self.succeeded_step is None:
                self.succeeded_step = self.elapsed
            return VerificationResult(
                decision=Decision.ADVANCE,
                reason=f"done_when satisfied for {self.contract.hold_steps} consecutive step(s): "
                       f"{self.contract.done_description}",
                hold=self.hold, elapsed=self.elapsed, predicates=dict(predicates),
            )

        if self.elapsed >= self.contract.max_steps:
            # never latched within budget -> TIMEOUT; planner should REPLAN.
            return VerificationResult(
                decision=Decision.REPLAN,
                reason=f"step budget {self.contract.max_steps} exhausted without done_when "
                       f"({self.contract.done_description})",
                hold=self.hold, elapsed=self.elapsed, predicates=dict(predicates),
            )

        return VerificationResult(
            decision=Decision.CONTINUE, reason="done_when not yet satisfied",
            hold=self.hold, elapsed=self.elapsed, predicates=dict(predicates),
        )
