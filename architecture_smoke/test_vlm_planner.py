"""Deterministic safety tests for the obs-only typed VLM planner."""
from __future__ import annotations

import json
import unittest

import bindings
from async_control import ControlResponse, RequestKind, ResponseStatus, assert_runtime_obs_only
from obs_verifier import ObsInput
from planner import VLMPlanner
from schemas import PlannerContext, SkillResult, SkillStatus
from skills import SkillRegistry


class ScriptedService:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []

    def complete(self, request, timeout_s):
        self.requests.append(request)
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        if callable(output):
            return output(request)
        return ControlResponse.from_request(request, payload={"text": output})


class VLMPlannerTests(unittest.TestCase):
    def setUp(self):
        self.registry = SkillRegistry(bindings.CONTRACTS)
        images = {name: [[[0, 0, 0]]] for name in (
            "robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")}
        self.obs = ObsInput(images=images, proprio=[0.0] * 14, step=7)

    def text(self, name, **overrides):
        payload = self.registry.typed_call(name).as_dict()
        payload.update(overrides)
        return json.dumps(payload)

    def ctx(self, result=None, budget=600, predicates=None):
        return PlannerContext(
            goal=bindings.GOAL,
            predicates=predicates or {},
            skill_catalog=self.registry.names,
            last_result=result,
            step_budget_remaining=budget,
            observation=self.obs,
        )

    @staticmethod
    def result(skill, status=SkillStatus.SUCCESS):
        return SkillResult(
            status=status, skill=skill, args={}, instruction="", steps=1,
            success_step=1 if status is SkillStatus.SUCCESS else None,
            terminated_by="test", reason="test", predicates={"official_check_success": True})

    def test_valid_plan_is_typed_and_uses_canonical_planner_envelope(self):
        service = ScriptedService([self.text("GRASP_OBJECT")])
        planner = VLMPlanner(self.registry, service, max_retries=0)
        call = planner.plan(self.ctx(predicates={"official_check_success": True}))
        self.assertEqual(call, self.registry.typed_call("GRASP_OBJECT"))
        request = service.requests[0]
        self.assertIs(request.request_kind, RequestKind.PLANNER)
        self.assertNotIn("predicates", request.payload)
        self.assertNotIn("official_check_success", repr(request.payload))
        assert_runtime_obs_only(request.payload)

    def test_unknown_skill_falls_back(self):
        bad = json.loads(self.text("GRASP_OBJECT"))
        bad["name"] = "TELEPORT_OBJECT"
        planner = VLMPlanner(self.registry, ScriptedService([json.dumps(bad)]), max_retries=0)
        call = planner.plan(self.ctx())
        self.assertEqual(call.name, "GRASP_OBJECT")
        self.assertIn("fallback", planner.rationale(call, None))

    def test_missing_args_malformed_json_and_excess_budget_fall_back(self):
        cases = [
            "not-json",
            self.text("GRASP_OBJECT", args={"object": "blender_lid"}),
            self.text("GRASP_OBJECT", budget=209),
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                planner = VLMPlanner(self.registry, ScriptedService([raw]), max_retries=0)
                self.assertEqual(planner.plan(self.ctx()).name, "GRASP_OBJECT")
                self.assertIn("fallback", planner.rationale(None, None))

    def test_timeout_falls_back_after_bounded_retry(self):
        service = ScriptedService([TimeoutError("slow"), TimeoutError("slow")])
        planner = VLMPlanner(self.registry, service, max_retries=1)
        self.assertEqual(planner.plan(self.ctx()).name, "GRASP_OBJECT")
        self.assertEqual(len(service.requests), 2)
        self.assertIn("TimeoutError", planner.rationale(None, None))

    def test_stale_response_falls_back(self):
        def stale(request):
            return ControlResponse(
                request.episode_id, request.skill_id, request.observation_step,
                "different-id", request.request_kind, payload={"text": self.text("GRASP_OBJECT")})
        planner = VLMPlanner(self.registry, ScriptedService([stale]), max_retries=0)
        self.assertEqual(planner.plan(self.ctx()).name, "GRASP_OBJECT")
        self.assertIn("stale planner response", planner.rationale(None, None))

    def test_repeated_cycle_is_guarded(self):
        service = ScriptedService([self.text("GRASP_OBJECT"), self.text("GRASP_OBJECT")])
        planner = VLMPlanner(
            self.registry, service, max_retries=0, max_cycle_repeats=1)
        self.assertEqual(planner.plan(self.ctx()).name, "GRASP_OBJECT")
        second = planner.plan(self.ctx())
        self.assertEqual(second.name, "GRASP_OBJECT")
        self.assertIn("repeated planner cycle", planner.rationale(second, None))

    def test_obs_only_grasp_move_place_typed_smoke(self):
        service = ScriptedService([
            self.text("GRASP_OBJECT"), self.text("MOVE_OBJECT"), self.text("PLACE_OBJECT")])
        planner = VLMPlanner(self.registry, service, max_retries=0)
        calls = []
        result = None
        for expected in ("GRASP_OBJECT", "MOVE_OBJECT", "PLACE_OBJECT"):
            call = planner.plan(self.ctx(result=result))
            calls.append(call.name)
            self.registry.validate_call(call)
            result = self.result(expected)
        self.assertEqual(calls, ["GRASP_OBJECT", "MOVE_OBJECT", "PLACE_OBJECT"])
        for request in service.requests:
            assert_runtime_obs_only(request.payload)
            self.assertNotIn("official_check_success", repr(request.payload))

    def test_service_error_status_falls_back(self):
        def error(request):
            return ControlResponse.from_request(
                request, status=ResponseStatus.ERROR, error="model failed")
        planner = VLMPlanner(self.registry, ScriptedService([error]), max_retries=0)
        self.assertEqual(planner.plan(self.ctx()).name, "GRASP_OBJECT")
        self.assertIn("model failed", planner.rationale(None, None))


if __name__ == "__main__":
    unittest.main()
