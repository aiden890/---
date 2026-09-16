"""ExecutionManager -- the skill-execution state machine.

Orchestrates, for ONE SkillCall:
    validate can_start -> render instruction -> [ base VLA infer -> execute a
    short action window -> verifier.update ] loop -> SkillResult.

Owns step budgets and the retry/replan transition mapping. Contains NO
task-specific predicate logic (that lives in the SkillContract via the verifier)
and NO next-skill selection (that is the planner's job). It is generic over any
registry/policy/environment/verifier that honor the schema interfaces.

A "short action window" here = one environment step; the base VLA re-plans a
fresh chunk every ``replan_steps`` steps (verbatim skill_eval semantics). The
verifier is consulted after every executed step and owns termination/handoff.
"""
from __future__ import annotations

from schemas import (AdapterMode, Decision, SkillCall, SkillResult, SkillStatus,
                     VerificationResult)
from verifier import PredicateVerifier


class ExecutionManager:
    def __init__(self, registry, policy, environment, trace, adapter_mode=AdapterMode.DISABLED):
        self.registry = registry
        self.policy = policy
        self.env = environment
        self.trace = trace
        self.adapter_mode = adapter_mode

    def execute(self, call: SkillCall) -> SkillResult:
        contract = self.registry.validate_call(call)          # schema validation
        instruction = contract.render_instruction(call.args)  # fixed NL render
        verifier = PredicateVerifier(contract)

        pred = self.env.predicates()
        can_start = verifier.check_can_start(pred)
        self.trace.route(call.name, self.adapter_mode.value, None, contract.max_steps, can_start)
        if not can_start:
            # precondition unmet -> FAILED, planner may RETRY/REPLAN.
            return self._result(SkillStatus.FAILED, call, instruction, 0, None,
                                "can_start_failed", "can_start precondition not satisfied", pred)

        action_plan = []
        last_v: VerificationResult | None = None
        steps = 0
        while steps < contract.max_steps:
            if not action_plan:
                pin = self.env.build_policy_input(instruction, self.adapter_mode, None)
                out = self.policy.infer(pin)
                # hard invariant: base policy only, no adapter loaded.
                assert out.adapter_checkpoint is None, "adapter checkpoint leaked into policy output"
                assert out.adapter_mode in (AdapterMode.DISABLED, AdapterMode.BASE_ONLY)
                action_plan = list(out.action_chunk)
                self.trace.window(call.name, steps, out.chunk_len, len(action_plan),
                                  out.adapter_mode.value, "EXECUTE", verifier.hold,
                                  _digest(pred))
            action = action_plan.pop(0)
            _, done, trunc, _ = self.env.step(action)
            steps += 1
            pred = self.env.predicates()
            v = verifier.update(pred)
            last_v = v
            fidx = self.env.maybe_record_frame(force=(v.decision is Decision.ADVANCE or done or trunc))
            if fidx is not None:
                self.trace.frame(call.name, steps, fidx)

            if v.decision is Decision.ADVANCE:
                self.trace.verify(call.name, v.decision.value, v.reason, v.elapsed, v.hold)
                return self._result(SkillStatus.SUCCESS, call, instruction, steps,
                                    verifier.succeeded_step, "success", v.reason, pred)
            if v.decision is Decision.REPLAN:
                self.trace.verify(call.name, v.decision.value, v.reason, v.elapsed, v.hold)
                return self._result(SkillStatus.TIMEOUT, call, instruction, steps, None,
                                    "timeout", v.reason, pred)
            if done or trunc:
                self.trace.verify(call.name, "TERMINATED", f"env done={done} trunc={trunc}",
                                  v.elapsed, v.hold)
                return self._result(SkillStatus.FAILED, call, instruction, steps, None,
                                    "env_terminated", f"env done={done} trunc={trunc}", pred)

        reason = last_v.reason if last_v else "budget exhausted"
        self.trace.verify(call.name, Decision.REPLAN.value, reason,
                          last_v.elapsed if last_v else steps, last_v.hold if last_v else 0)
        return self._result(SkillStatus.TIMEOUT, call, instruction, steps, None,
                            "timeout", reason, pred)

    def _result(self, status, call, instruction, steps, success_step, terminated_by, reason, pred):
        return SkillResult(status=status, skill=call.name, args=dict(call.args),
                           instruction=instruction, steps=steps, success_step=success_step,
                           terminated_by=terminated_by, reason=reason, predicates=dict(pred))


_DIGEST_KEYS = ("lid_grasped", "in_preplace_region", "lid_on_blender",
                "official_check_success", "lid_xy_to_closed_pos", "eef_lid_dist")


def _digest(p):
    return {k: p.get(k) for k in _DIGEST_KEYS if k in p}
