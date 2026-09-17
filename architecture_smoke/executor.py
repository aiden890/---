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


# Map the generic skill contract names to the obs_verifier question keys.
_SKILL_QUESTION_KEY = {
    "GRASP_OBJECT": "GRASP_OBJECT",
    "MOVE_OBJECT": "MOVE_OBJECT",
    "PLACE_OBJECT": "PLACE_OBJECT",
}


class ExecutionManager:
    def __init__(self, registry, policy, environment, trace, adapter_mode=AdapterMode.DISABLED,
                 vlm_backend=None, vlm_min_interval=16, hysteresis_k=2, tau=0.6,
                 event_gated=True):
        self.registry = registry
        self.policy = policy
        self.env = environment
        self.trace = trace
        self.adapter_mode = adapter_mode
        # When a VLM backend is supplied, skill termination is judged obs-only by
        # the policy's own frozen Qwen3-VL (ObsVLMVerifier); otherwise the parent
        # privileged-predicate PredicateVerifier is used.
        self.vlm_backend = vlm_backend
        self.vlm_min_interval = vlm_min_interval
        self.hysteresis_k = hysteresis_k
        self.tau = tau
        self.event_gated = event_gated

    def execute(self, call: SkillCall) -> SkillResult:
        if self.vlm_backend is not None:
            return self._execute_vlm(call)
        return self._execute_predicate(call)

    # ------------------------------------------------------------------ #
    #  obs-only VLM verifier path (Qwen3-VL VQA, proprio-gated)            #
    # ------------------------------------------------------------------ #
    def _execute_vlm(self, call: SkillCall) -> SkillResult:
        from obs_verifier import ObsVLMVerifier, SKILL_QUESTIONS
        from success_gate import make_success_gate

        contract = self.registry.validate_call(call)
        instruction = contract.render_instruction(call.args)
        qkey = _SKILL_QUESTION_KEY.get(call.name, call.name)
        verifier = ObsVLMVerifier(
            qkey, self.vlm_backend, max_steps=contract.max_steps,
            question_text=SKILL_QUESTIONS[qkey],
            vlm_min_interval=self.vlm_min_interval, hysteresis_k=self.hysteresis_k,
            tau=self.tau, event_gated=self.event_gated)
        # SECOND ROLE: strict, view-routed, episode-level success judge, SEPARATE
        # from the boundary latch above. It accumulates per-frame P(yes) on the
        # recorded-frame cadence and renders a strict pass/fail at skill end.
        # None for skills without a calibrated gate (e.g. MOVE).
        success_gate = make_success_gate(call.name, self.vlm_backend)

        # can_start stays obs-agnostic here: the sequential planner only issues a
        # skill when it is its turn, so we always start (no privileged precondition).
        self.trace.route(call.name, self.adapter_mode.value, None, contract.max_steps, True)

        action_plan = []
        last_v: VerificationResult | None = None
        steps = 0
        while steps < contract.max_steps:
            if not action_plan:
                pin = self.env.build_policy_input(instruction, self.adapter_mode, None)
                out = self.policy.infer(pin)
                assert out.adapter_checkpoint is None, "adapter checkpoint leaked into policy output"
                assert out.adapter_mode in (AdapterMode.DISABLED, AdapterMode.BASE_ONLY)
                action_plan = list(out.action_chunk)
                self.trace.window(call.name, steps, out.chunk_len, len(action_plan),
                                  out.adapter_mode.value, "EXECUTE", verifier.consecutive_yes,
                                  {"judge": "obs_vlm"})
            action = action_plan.pop(0)
            _, done, trunc, _ = self.env.step(action)
            steps += 1

            obs = self.env.obs_for_verifier()          # obs-only: 3-cam + 14D proprio
            v = verifier.update(obs)
            last_v = v
            diag = v.predicates
            queried = bool(diag.get("vlm_queried_this_step"))
            # record a frame on a VLM query, on ADVANCE, or on env end, so the
            # overlay shows every judgement moment.
            fidx = self.env.maybe_record_frame(
                force=(queried or v.decision is Decision.ADVANCE or done or trunc))
            if fidx is not None:
                self.trace.frame(call.name, steps, fidx)
                # strict success judge observes the SAME recorded frames (cheap
                # cadence); it is view-routed and episode-level, independent of
                # the boundary latch's decision.
                if success_gate is not None:
                    try:
                        success_gate.observe(obs.images if hasattr(obs, "images") else obs)
                    except Exception:  # noqa: BLE001 -- never let the gate break the loop
                        pass
                if queried or v.decision is Decision.ADVANCE:
                    self.trace.vlm(call.name, steps, fidx, diag.get("vlm_prob"),
                                   diag.get("consecutive_yes"), queried,
                                   v.decision.value, diag.get("proprio_candidate_stop"),
                                   diag.get("proprio_gripper_closed"))

            if v.decision is Decision.ADVANCE:
                self.trace.verify(call.name, v.decision.value, v.reason, v.elapsed, v.hold)
                res = self._result(SkillStatus.SUCCESS, call, instruction, steps,
                                   verifier.succeeded_step, "vlm_success", v.reason,
                                   self.env.predicates())
                res.vlm_stats = verifier.stats()
                # attach the strict success verdict (obs-only), SEPARATE from the
                # boundary ADVANCE above -- the report reader can now see a skill
                # that advanced but did NOT pass the strict success gate.
                if success_gate is not None:
                    res.success_gate = success_gate.verdict()
                return res
            if v.decision is Decision.REPLAN:
                self.trace.verify(call.name, v.decision.value, v.reason, v.elapsed, v.hold)
                res = self._result(SkillStatus.TIMEOUT, call, instruction, steps, None,
                                   "vlm_timeout", v.reason, self.env.predicates())
                res.vlm_stats = verifier.stats()
                # strict success verdict is meaningful even when the boundary
                # latch never fired (timeout): the window mean of the accumulated
                # frames says whether the end state actually looks successful.
                if success_gate is not None:
                    res.success_gate = success_gate.verdict()
                return res
            if done or trunc:
                self.trace.verify(call.name, "TERMINATED", f"env done={done} trunc={trunc}",
                                  v.elapsed, v.hold)
                res = self._result(SkillStatus.FAILED, call, instruction, steps, None,
                                   "env_terminated", f"env done={done} trunc={trunc}",
                                   self.env.predicates())
                res.vlm_stats = verifier.stats()
                if success_gate is not None:
                    res.success_gate = success_gate.verdict()
                return res

        reason = last_v.reason if last_v else "budget exhausted"
        self.trace.verify(call.name, Decision.REPLAN.value, reason,
                          last_v.elapsed if last_v else steps, last_v.hold if last_v else 0)
        res = self._result(SkillStatus.TIMEOUT, call, instruction, steps, None,
                           "vlm_timeout", reason, self.env.predicates())
        res.vlm_stats = verifier.stats()
        if success_gate is not None:
            res.success_gate = success_gate.verdict()
        return res

    # ------------------------------------------------------------------ #
    #  privileged-predicate path (original behaviour)                     #
    # ------------------------------------------------------------------ #
    def _execute_predicate(self, call: SkillCall) -> SkillResult:
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
