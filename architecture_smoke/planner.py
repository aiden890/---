"""Scripted compatibility planners and the observation-only VLM planner.

IMPORTANT: this smoke test ships an ORACLE / SCRIPTED planner stub, NOT a
learned VLM planner. No skill-conditioned VLM planner has been trained yet;
the architecture contract only requires that the planner *interface* is honored
(input PlannerContext -> output a schema-valid SkillCall, no policy/sim access).
The stub selects the next skill from the current predicate state using the same
subgoal ordering a competent planner would produce for CloseBlenderLid. Never
report its output as a learned planner.

The class only depends on schemas + the SkillRegistry catalog; it makes no
simulator or policy calls and holds no environment handle.
"""
from __future__ import annotations

import json
from typing import Protocol

from async_control import ControlRequest, ControlResponse, RequestKind, ResponseStatus
from schemas import PlannerContext, SkillCall, SkillResult, SkillStatus
from skills import SkillRegistry


class OraclePlanner:
    """Scripted planner stub honoring the planner interface.

    Policy (CloseBlenderLid subgoal chain):
      not grasped              -> GRASP_OBJECT
      grasped, not in preplace -> MOVE_OBJECT
      grasped, in preplace     -> PLACE_OBJECT
      task predicate satisfied -> None (episode done)

    On a FAILED/TIMEOUT last result the planner re-issues the same subgoal
    (retry/replan) rather than skipping ahead -- demonstrating the planner can
    advance OR retry without manual intervention.
    """

    kind = "oracle_stub"  # provenance tag written into every trace record

    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def plan(self, ctx: PlannerContext) -> SkillCall | None:
        p = ctx.predicates
        if p.get("official_check_success"):
            return None  # task complete -> terminal

        grasped = bool(p.get("lid_grasped"))
        in_preplace = bool(p.get("in_preplace_region"))

        if not grasped:
            name = "GRASP_OBJECT"
        elif not in_preplace:
            name = "MOVE_OBJECT"
        else:
            name = "PLACE_OBJECT"
        return self.registry.typed_call(name)

    @staticmethod
    def rationale(call: SkillCall | None, last: SkillResult | None) -> str:
        if call is None:
            return "task predicate satisfied; terminate episode"
        base = f"selected {call.name} from current predicate state"
        if last is not None and last.status is not SkillStatus.SUCCESS:
            return f"{base} (retry after previous {last.skill}={last.status.value})"
        return base


class SequentialPlanner:
    """Obs-only planner: advances GRASP_OBJECT -> MOVE_OBJECT -> PLACE_OBJECT on the
    verifier's SUCCESS, NOT on any privileged simulator predicate.

    Used with the obs-only VLM verifier so the ENTIRE runtime (skill termination
    AND next-skill selection) reads only what the robot receives: the VLM verifier
    ADVANCE (obs P(yes) latch) drives progression; a FAILED/TIMEOUT re-issues the
    same skill up to ``max_retries`` before moving on. It never queries
    ``ctx.predicates`` (the sim ground-truth), so the pipeline is transferable.

    This is still a SCRIPTED stub (fixed subgoal order), NOT a learned planner.
    """

    kind = "sequential_obs_stub"
    SEQUENCE = ("GRASP_OBJECT", "MOVE_OBJECT", "PLACE_OBJECT")

    def __init__(self, registry: SkillRegistry, max_retries: int = 1):
        self.registry = registry
        self.max_retries = max_retries
        self._idx = 0
        self._retries = 0
        self._last_seen = None

    def plan(self, ctx: PlannerContext) -> SkillCall | None:
        last = ctx.last_result
        # advance/retry based ONLY on the obs-only verifier outcome
        if last is not None and last is not self._last_seen:
            self._last_seen = last
            if last.status is SkillStatus.SUCCESS:
                self._idx += 1
                self._retries = 0
            else:
                self._retries += 1
                if self._retries > self.max_retries:
                    self._idx += 1        # give up on this skill, move on honestly
                    self._retries = 0
        if self._idx >= len(self.SEQUENCE):
            return None                    # all skills attempted -> episode done
        name = self.SEQUENCE[self._idx]
        return self.registry.typed_call(name)

    @staticmethod
    def rationale(call: SkillCall | None, last: SkillResult | None) -> str:
        if call is None:
            return "all skills attempted (obs-only verifier); terminate episode"
        base = f"sequential obs-only plan -> {call.name}"
        if last is not None and last.status is not SkillStatus.SUCCESS:
            return f"{base} (retry after {last.skill}={last.status.value})"
        return base


class VLMPlannerService(Protocol):
    """Low-priority shared-model service used by :class:`VLMPlanner`."""

    def complete(self, request: ControlRequest, timeout_s: float) -> ControlResponse:
        """Return JSON text in ``response.payload['text']`` for one planner request."""
        raise NotImplementedError


class PlannerOutputError(ValueError):
    """A model response was not a canonical, registry-backed SkillCall."""


class VLMPlanner:
    """Observation-only structured planner with deterministic safe fallback."""

    kind = "vlm_obs_typed"
    _FIELDS = frozenset(("name", "args", "instruction", "contract", "budget"))

    def __init__(self, registry: SkillRegistry, service: VLMPlannerService, *,
                 episode_id: str = "episode", timeout_s: float = 5.0,
                 max_retries: int = 1, max_plan_length: int = 12,
                 max_cycle_repeats: int = 2, fallback_retries: int = 1):
        if timeout_s <= 0 or max_retries < 0 or max_plan_length <= 0 or max_cycle_repeats <= 0:
            raise ValueError("invalid VLM planner bounds")
        self.registry = registry
        self.service = service
        self.episode_id = str(episode_id)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.max_plan_length = int(max_plan_length)
        self.max_cycle_repeats = int(max_cycle_repeats)
        self.fallback = SequentialPlanner(registry, max_retries=fallback_retries)
        self._request_seq = 0
        self._accepted = 0
        self._signature_counts = {}
        self._history = []
        self._last_result = None
        self._rationale = "planner not called"

    def plan(self, ctx: PlannerContext) -> SkillCall | None:
        # Keep fallback state synchronized even on successful VLM calls, so a
        # later model fault resumes from the deterministic expected subgoal.
        fallback_call = self.fallback.plan(ctx)
        if fallback_call is not None and fallback_call.budget is not None:
            fallback_call = self.registry.typed_call(
                fallback_call.name, fallback_call.args,
                budget=min(fallback_call.budget, ctx.step_budget_remaining))
        self._record_result(ctx.last_result)
        if self._accepted >= self.max_plan_length:
            self._rationale = "max plan length reached; terminate safely"
            return None
        try:
            observation, step = self._validate_observation(ctx.observation)
        except Exception as exc:  # noqa: BLE001 -- malformed observation is a safe fallback
            return self._fallback(fallback_call, f"invalid observation: {exc}")

        prompt = self._prompt(ctx)
        last_error = "planner service failed"
        for attempt in range(self.max_retries + 1):
            self._request_seq += 1
            request = ControlRequest(
                episode_id=self.episode_id,
                skill_id=(ctx.last_result.skill if ctx.last_result else "PLANNER"),
                observation_step=step,
                request_id=f"{self.episode_id}:planner:{self._request_seq}",
                request_kind=RequestKind.PLANNER,
                payload={"goal": ctx.goal, "observation": observation,
                         "history": list(self._history), "prompt": prompt})
            try:
                response = self.service.complete(request, self.timeout_s)
                if not isinstance(response, ControlResponse):
                    raise TypeError("planner service must return ControlResponse")
                if response.identity != request.identity:
                    raise PlannerOutputError("stale planner response identity")
                if response.status is not ResponseStatus.OK:
                    raise PlannerOutputError(response.error or response.status.value)
                call = self._parse(response.payload.get("text"), ctx.step_budget_remaining)
                signature = (call.name, tuple(sorted(call.args.items())))
                if self._signature_counts.get(signature, 0) >= self.max_cycle_repeats:
                    raise PlannerOutputError("repeated planner cycle")
                self._signature_counts[signature] = self._signature_counts.get(signature, 0) + 1
                self._accepted += 1
                self._history.append({"type": "plan", "call": call.as_dict()})
                self._rationale = f"validated obs-only VLM SkillCall (attempt {attempt + 1})"
                return call
            except Exception as exc:  # noqa: BLE001 -- model/RPC errors must not reach control
                last_error = f"{type(exc).__name__}: {exc}"
        return self._fallback(fallback_call, last_error)

    def _record_result(self, result: SkillResult | None) -> None:
        if result is None or result is self._last_result:
            return
        self._last_result = result
        # Omit result.predicates: it may contain simulator ground truth.
        self._history.append({
            "type": "result", "status": result.status.value, "skill": result.skill,
            "steps": result.steps, "terminated_by": result.terminated_by,
            "reason": result.reason,
        })

    @staticmethod
    def _validate_observation(observation):
        if observation is None or not hasattr(observation, "images") or not hasattr(observation, "proprio"):
            raise ValueError("expected images + proprio")
        images = observation.images
        if not isinstance(images, dict) or len(images) != 3:
            raise ValueError("expected exactly three camera views")
        if len(observation.proprio) != 14:
            raise ValueError("expected proprio14")
        return observation, int(getattr(observation, "step", 0))

    def _prompt(self, ctx: PlannerContext) -> str:
        schema = {"name": "string", "args": {"required_arg": "string"},
                  "instruction": "registry-rendered string",
                  "contract": "registry done_when string", "budget": "positive integer"}
        return (
            "Choose exactly one next robot skill from the registry. Use only the supplied "
            "goal, three camera views, proprio14, and history. Never infer or request simulator "
            "predicates or official success labels. Return one JSON object and no prose.\n" +
            json.dumps({"schema": schema, "skill_registry": self.registry.catalog(),
                        "goal": ctx.goal, "history": self._history}, sort_keys=True))

    def _parse(self, raw, remaining_budget: int) -> SkillCall:
        if not isinstance(raw, str):
            raise PlannerOutputError("planner output must be JSON text")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PlannerOutputError(f"invalid JSON: {exc.msg}") from exc
        if not isinstance(data, dict) or set(data) != self._FIELDS:
            raise PlannerOutputError(f"SkillCall fields must be exactly {sorted(self._FIELDS)}")
        if not isinstance(data["name"], str) or not isinstance(data["args"], dict):
            raise PlannerOutputError("name must be string and args must be object")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in data["args"].items()):
            raise PlannerOutputError("all SkillCall args must be strings")
        if not isinstance(data["instruction"], str) or not isinstance(data["contract"], str):
            raise PlannerOutputError("instruction and contract must be strings")
        contract = self.registry.get(data["name"])
        if set(data["args"]) != set(contract.arg_schema):
            raise PlannerOutputError(
                f"SkillCall({data['name']}) args must be exactly {list(contract.arg_schema)}")
        call = SkillCall(**data)
        self.registry.validate_call(call)
        assert call.budget is not None
        if call.budget > remaining_budget:
            raise PlannerOutputError(
                f"SkillCall({call.name}) budget {call.budget} exceeds episode remainder {remaining_budget}")
        return call

    def _fallback(self, call: SkillCall | None, reason: str) -> SkillCall | None:
        self._rationale = f"deterministic SequentialPlanner fallback: {reason}"
        if call is not None:
            self._history.append({"type": "fallback", "reason": reason,
                                  "call": call.as_dict()})
        return call

    def rationale(self, call: SkillCall | None, last: SkillResult | None) -> str:
        return self._rationale
