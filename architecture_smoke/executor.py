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

import itertools

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
                 event_gated=True, verifier_operating_points=None,
                 synchronous_verifier=False, episode_id="episode", verifier_timeout_s=5.0):
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
        self.verifier_operating_points = verifier_operating_points or {}
        # The target/default path is latest-only asynchronous control.  The old
        # blocking implementation remains behind one explicit compatibility flag.
        self.synchronous_verifier = bool(synchronous_verifier)
        self.episode_id = str(episode_id)
        self.verifier_timeout_s = float(verifier_timeout_s)
        self._request_ids = itertools.count(1)

    def execute(self, call: SkillCall) -> SkillResult:
        if self.vlm_backend is not None:
            if not self.synchronous_verifier:
                return self._execute_vlm_async(call)
            return self._execute_vlm(call)
        return self._execute_predicate(call)

    def _execute_vlm_async(self, call: SkillCall) -> SkillResult:
        """Execute actions while verifier/endpoint VLM work runs in the background."""
        from async_control import (AsyncVerifierClient, ControlRequest, ControlResponse,
                                   RequestKind, ResponseStatus)
        from obs_verifier import ObsVLMVerifier, SKILL_QUESTIONS
        from success_gate import make_success_gate

        contract = self.registry.validate_call(call)
        instruction = contract.render_instruction(call.args)
        qkey = _SKILL_QUESTION_KEY.get(call.name, call.name)
        op = self.verifier_operating_points.get(call.name, {})
        verifier = ObsVLMVerifier(
            qkey, self.vlm_backend, max_steps=contract.max_steps,
            question_text=SKILL_QUESTIONS[qkey],
            vlm_min_interval=op.get("vlm_min_interval", self.vlm_min_interval),
            hysteresis_k=op.get("hysteresis_k", self.hysteresis_k),
            tau=op.get("tau", self.tau), event_gated=self.event_gated,
            view=op.get("view", "full"), sequence_model=op.get("sequence_model"))
        success_gate = make_success_gate(call.name, self.vlm_backend)

        def transport(request, timeout_s):
            # RemoteVLMScorerBackend/socket implementations own the actual socket
            # deadline.  The worker serialises this entire service: max one forward.
            if hasattr(self.vlm_backend, "set_timeout"):
                self.vlm_backend.set_timeout(timeout_s)
            set_control_request = getattr(self.vlm_backend, "set_control_request", None)
            if set_control_request is not None:
                set_control_request(request)
            obs = request.payload["observation"]
            if request.request_kind is RequestKind.BOUNDARY:
                result = verifier.update(obs)
                # Accumulate obs-only evidence here; only ENDPOINT renders/uses the
                # strict gate verdict, and only after a boundary candidate-stop.
                if success_gate is not None:
                    try:
                        success_gate.observe(obs.images if hasattr(obs, "images") else obs)
                    except Exception:  # noqa: BLE001 -- gate failure cannot stop control
                        pass
                recommendation = result.decision.value
                payload = {
                    "reason": result.reason, "elapsed": result.elapsed, "hold": result.hold,
                    "candidate_stop": result.decision is Decision.ADVANCE,
                    "diagnostics": dict(result.predicates),
                }
                # A remote boundary candidate never advances the skill directly.
                if result.decision is Decision.ADVANCE:
                    recommendation = Decision.CONTINUE.value
                return ControlResponse.from_request(
                    request, recommendation=recommendation, payload=payload)

            if request.request_kind is RequestKind.ENDPOINT:
                verdict = (success_gate.verdict() if success_gate is not None
                           else {"success": True, "reason": "no_strict_gate"})
                recommendation = (Decision.ADVANCE.value if verdict.get("success")
                                  else Decision.CONTINUE.value)
                return ControlResponse.from_request(
                    request, recommendation=recommendation,
                    payload={"success_gate": verdict, "reason": verdict.get("reason")})
            return ControlResponse.from_request(
                request, recommendation=Decision.CONTINUE.value,
                payload={"reason": "planner response not consumed by skill executor"})

        def telemetry(event, **payload):
            if hasattr(self.trace, "control"):
                self.trace.control(event, **payload)

        client = AsyncVerifierClient(
            transport, timeout_s=self.verifier_timeout_s, telemetry=telemetry)
        client.set_context(self.episode_id, call.name)
        self.trace.route(call.name, self.adapter_mode.value, None, contract.max_steps, True)

        action_plan = []
        steps = 0
        endpoint_outstanding = False
        last_obs = None
        last_reason = "budget exhausted"
        last_gate = None

        def submit(kind, obs, step):
            rid = f"{self.episode_id}:{call.name}:{next(self._request_ids)}"
            client.submit(ControlRequest(
                episode_id=self.episode_id, skill_id=call.name,
                observation_step=step, request_id=rid, request_kind=kind,
                payload={"observation": obs, "instruction": instruction}))

        def consume():
            nonlocal endpoint_outstanding, last_reason, last_gate
            for response in client.poll():
                if response.status is not ResponseStatus.OK:
                    last_reason = response.error or response.status.value
                    continue
                # Manager is the sole transition authority: validate and map the
                # advisory string here rather than allowing transport/server state.
                try:
                    decision = Decision(response.recommendation)
                except (TypeError, ValueError):
                    last_reason = "invalid remote recommendation"
                    continue
                last_reason = str(response.payload.get("reason", decision.value))
                if response.request_kind is RequestKind.BOUNDARY:
                    diag = response.payload.get("diagnostics", {})
                    self.trace.verify(call.name, decision.value, last_reason,
                                      response.payload.get("elapsed", 0),
                                      response.payload.get("hold", 0))
                    if response.payload.get("candidate_stop"):
                        endpoint_outstanding = True
                        submit(RequestKind.ENDPOINT, last_obs, steps)
                    elif decision in (Decision.RETRY, Decision.REPLAN):
                        return decision
                    if diag.get("vlm_queried_this_step"):
                        self.trace.vlm(call.name, response.observation_step, None,
                                       diag.get("vlm_prob"), diag.get("consecutive_yes"), True,
                                       decision.value, diag.get("proprio_candidate_stop"),
                                       diag.get("proprio_gripper_closed"))
                elif response.request_kind is RequestKind.ENDPOINT:
                    endpoint_outstanding = False
                    last_gate = response.payload.get("success_gate")
                    if decision is Decision.ADVANCE:
                        return Decision.ADVANCE
            return Decision.CONTINUE

        try:
            while steps < contract.max_steps:
                transition = consume()
                if transition is Decision.ADVANCE:
                    result = self._result(SkillStatus.SUCCESS, call, instruction, steps, steps,
                                          "async_endpoint_success", last_reason, {})
                    result.vlm_stats = verifier.stats()
                    result.success_gate = last_gate
                    return result
                if transition is Decision.REPLAN:
                    result = self._result(SkillStatus.TIMEOUT, call, instruction, steps, None,
                                          "async_vlm_replan", last_reason, {})
                    result.vlm_stats = verifier.stats()
                    result.success_gate = last_gate
                    return result
                if transition is Decision.RETRY:
                    result = self._result(SkillStatus.FAILED, call, instruction, steps, None,
                                          "async_vlm_retry", last_reason, {})
                    result.vlm_stats = verifier.stats()
                    return result
                if not action_plan:
                    pin = self.env.build_policy_input(instruction, self.adapter_mode, None)
                    out = self.policy.infer(pin)
                    assert out.adapter_checkpoint is None, "adapter checkpoint leaked into policy output"
                    assert out.adapter_mode in (AdapterMode.DISABLED, AdapterMode.BASE_ONLY)
                    action_plan = list(out.action_chunk)
                    self.trace.window(call.name, steps, out.chunk_len, len(action_plan),
                                      out.adapter_mode.value, "EXECUTE", 0,
                                      {"judge": "async_obs_vlm"})
                action = action_plan.pop(0)
                _, done, trunc, _ = self.env.step(action)
                steps += 1
                last_obs = self.env.obs_for_verifier()
                frame_index = self.env.maybe_record_frame(force=done or trunc)
                if frame_index is not None:
                    self.trace.frame(call.name, steps, frame_index)
                if not endpoint_outstanding:
                    submit(RequestKind.BOUNDARY, last_obs, steps)
                transition = consume()
                if transition is Decision.ADVANCE:
                    result = self._result(SkillStatus.SUCCESS, call, instruction, steps, steps,
                                          "async_endpoint_success", last_reason, {})
                    result.vlm_stats = verifier.stats()
                    result.success_gate = last_gate
                    return result
                if transition is Decision.REPLAN:
                    result = self._result(SkillStatus.TIMEOUT, call, instruction, steps, None,
                                          "async_vlm_replan", last_reason, {})
                    result.vlm_stats = verifier.stats()
                    result.success_gate = last_gate
                    return result
                if transition is Decision.RETRY:
                    result = self._result(SkillStatus.FAILED, call, instruction, steps, None,
                                          "async_vlm_retry", last_reason, {})
                    result.vlm_stats = verifier.stats()
                    return result
                if done or trunc:
                    result = self._result(SkillStatus.FAILED, call, instruction, steps, None,
                                          "env_terminated", f"env done={done} trunc={trunc}", {})
                    result.vlm_stats = verifier.stats()
                    return result
        finally:
            client.close(drain=False, timeout=self.verifier_timeout_s + 1.0)

        result = self._result(SkillStatus.TIMEOUT, call, instruction, steps, None,
                              "async_budget_exhausted", last_reason, {})
        result.vlm_stats = verifier.stats()
        result.success_gate = last_gate
        return result

    # ------------------------------------------------------------------ #
    #  obs-only VLM verifier path (Qwen3-VL VQA, proprio-gated)            #
    # ------------------------------------------------------------------ #
    def _execute_vlm(self, call: SkillCall) -> SkillResult:
        from obs_verifier import ObsVLMVerifier, SKILL_QUESTIONS
        from success_gate import make_success_gate

        contract = self.registry.validate_call(call)
        instruction = contract.render_instruction(call.args)
        qkey = _SKILL_QUESTION_KEY.get(call.name, call.name)
        op = self.verifier_operating_points.get(call.name, {})
        verifier = ObsVLMVerifier(
            qkey, self.vlm_backend, max_steps=contract.max_steps,
            question_text=SKILL_QUESTIONS[qkey],
            vlm_min_interval=op.get("vlm_min_interval", self.vlm_min_interval),
            hysteresis_k=op.get("hysteresis_k", self.hysteresis_k),
            tau=op.get("tau", self.tau), event_gated=self.event_gated,
            view=op.get("view", "full"),
            sequence_model=op.get("sequence_model"))
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
