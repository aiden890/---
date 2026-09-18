"""Fail-closed, one-call pre-episode full-plan planner.

The model returns only ``{"plan": [{"name": ..., "args": ...}, ...]}``.  This
module performs direct JSON parsing and uses the existing SkillRegistry to
produce canonical five-field SkillCalls.  It never repairs, retries, substitutes
skills, or falls back to a scripted planner.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

from async_control import ControlRequest, ControlResponse, RequestKind, ResponseStatus
from schemas import PlannerContext, SkillCall


class PlanRejected(ValueError):
    """The episode must execute zero actions because its plan is not trustworthy."""


class FullPlanService(Protocol):
    def complete(self, request: ControlRequest, timeout_s: float) -> ControlResponse:
        raise NotImplementedError


@dataclass(frozen=True)
class CanonicalPlan:
    calls: tuple[SkillCall, ...]
    reset_step: int
    reset_fingerprint: str


def observation_fingerprint(observation: Any) -> str:
    """Stable digest for detecting reset-snapshot divergence before action zero."""
    if observation is None or not hasattr(observation, "images") or not hasattr(observation, "proprio"):
        raise PlanRejected("reset observation must contain images and proprio")
    images = observation.images
    if not isinstance(images, dict) or len(images) != 3:
        raise PlanRejected("reset observation must contain exactly three camera views")
    digest = hashlib.sha256()
    for name in sorted(images):
        digest.update(name.encode("utf-8"))
        value = images[name]
        if hasattr(value, "tobytes"):
            digest.update(value.tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    proprio = observation.proprio
    if hasattr(proprio, "tobytes"):
        digest.update(proprio.tobytes())
    else:
        digest.update(repr(list(proprio)).encode("utf-8"))
    digest.update(str(int(getattr(observation, "step", 0))).encode("ascii"))
    return digest.hexdigest()


class PreEpisodePlanner:
    """Call a generic VLM exactly once and canonicalize one complete plan."""

    kind = "generic_qwen_pre_episode_full_plan"

    def __init__(self, registry, service: FullPlanService, *, episode_id: str = "episode",
                 timeout_s: float = 10.0):
        if timeout_s <= 0:
            raise ValueError("planner timeout must be positive")
        self.registry = registry
        self.service = service
        self.episode_id = str(episode_id)
        self.timeout_s = float(timeout_s)
        self.calls = 0
        self._planned = False
        self.last_error = None

    def plan_episode(self, ctx: PlannerContext) -> CanonicalPlan:
        if self._planned:
            raise PlanRejected("pre-episode planner may be called exactly once")
        self._planned = True
        self.calls += 1
        observation = ctx.observation
        fingerprint = observation_fingerprint(observation)
        step = int(getattr(observation, "step", 0))
        request = ControlRequest(
            episode_id=self.episode_id,
            skill_id="FULL_PLAN",
            observation_step=step,
            request_id=f"{self.episode_id}:pre-episode-plan",
            request_kind=RequestKind.PLANNER,
            payload={
                "goal": ctx.goal,
                "observation": observation,
                "prompt": self._prompt(ctx.goal),
            },
        )
        try:
            response = self.service.complete(request, self.timeout_s)
            if not isinstance(response, ControlResponse):
                raise PlanRejected("planner service must return ControlResponse")
            if response.identity != request.identity:
                raise PlanRejected("stale planner response identity")
            if response.status is not ResponseStatus.OK:
                raise PlanRejected(response.error or response.status.value)
            calls = self._canonicalize(response.payload.get("text"))
        except PlanRejected as exc:
            self.last_error = str(exc)
            raise
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise PlanRejected(self.last_error) from exc
        return CanonicalPlan(tuple(calls), step, fingerprint)

    def validate_execution_snapshot(self, plan: CanonicalPlan, observation: Any) -> None:
        step = int(getattr(observation, "step", 0))
        if step != plan.reset_step:
            raise PlanRejected("reset observation is stale before execution")
        if observation_fingerprint(observation) != plan.reset_fingerprint:
            raise PlanRejected("reset observation diverged before execution")

    def _prompt(self, goal: str) -> str:
        catalog = [
            {"name": name, "description": self.registry.get(name).description,
             "allowed_args": {key: [value] for key, value in
                              self.registry.get(name).default_args.items()}}
            for name in self.registry.names
        ]
        return (
            "Decide the complete robot skill sequence needed for the task visible in the "
            "initial camera observation. Use only the task instruction, image, and supplied "
            "registry. Return exactly one JSON object and no prose. Every step must contain "
            "exactly name and args; copy each argument value exactly from allowed_args. "
            "Do not emit instruction, contract, budget, simulator predicates, state labels, "
            "or unstated skills.\n" + json.dumps({
                "task_instruction": goal, "skill_registry": catalog,
                "output_schema": {"plan": [{"name": "registry skill name",
                                               "args": {"required_arg": "exact allowed value"}}]},
            }, sort_keys=True)
        )

    def _canonicalize(self, raw: Any) -> list[SkillCall]:
        if not isinstance(raw, str):
            raise PlanRejected("planner output must be JSON text")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PlanRejected(f"invalid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict) or set(payload) != {"plan"}:
            raise PlanRejected("full plan fields must be exactly ['plan']")
        steps = payload["plan"]
        if not isinstance(steps, list) or len(steps) != len(self.registry.names):
            raise PlanRejected("full plan length must match the skill registry")
        expected = self.registry.names
        calls = []
        for index, step in enumerate(steps):
            if not isinstance(step, dict) or set(step) != {"name", "args"}:
                raise PlanRejected(f"plan step {index} fields must be exactly ['args', 'name']")
            if step["name"] != expected[index]:
                raise PlanRejected("full plan sequence does not match registry order")
            args = step["args"]
            if not isinstance(args, dict) or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in args.items()):
                raise PlanRejected(f"plan step {index} args must be a string map")
            try:
                if set(args) != set(self.registry.get(step["name"]).arg_schema):
                    raise PlanRejected(f"plan step {index} argument keys do not match registry")
                calls.append(self.registry.typed_call(step["name"], args))
            except (KeyError, TypeError, ValueError) as exc:
                raise PlanRejected(f"invalid plan step {index}: {exc}") from exc
        return calls
