"""Production pre-episode planner and deterministic executor contract tests."""
from __future__ import annotations

import copy
import json
import unittest

import bindings
from async_control import ControlResponse, ResponseStatus
from episode import run_preplanned_episode
from obs_verifier import ObsInput
from pre_episode_planner import PreEpisodePlanner
from runtime_contract import (ACTION_JITTER_DEADLINE_MS, ACTION_RATE_HZ,
                              CHUNK_DEADLINE_MS, MAX_OBSERVATION_AGE_MS,
                              MAX_QUEUE_CHUNKS, POLICY_CHUNK_ACTIONS)
from schemas import SkillResult, SkillStatus
from skills import SkillRegistry


class Service:
    def __init__(self, output, *, stale=False, status=ResponseStatus.OK):
        self.output = output
        self.stale = stale
        self.status = status
        self.requests = []

    def complete(self, request, timeout_s):
        self.requests.append(request)
        if isinstance(self.output, BaseException):
            raise self.output
        response = ControlResponse.from_request(
            request, status=self.status, payload={"text": self.output},
            error=("planner error" if self.status is not ResponseStatus.OK else None))
        if self.stale:
            response = ControlResponse(
                response.episode_id, response.skill_id, response.observation_step,
                "stale-request", response.request_kind, status=response.status,
                payload=response.payload, error=response.error)
        return response


class Environment:
    def __init__(self, *, diverge=False):
        self.seed = 0
        self.actions = 0
        self._reads = 0
        self.diverge = diverge
        self.obs = ObsInput(
            images={name: [[[index, 0, 0]]] for index, name in enumerate((
                "robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"))},
            proprio=[0.0] * 14, step=0)

    def reset(self):
        self.actions = 0
        self._reads = 0
        return self.obs

    def obs_for_verifier(self):
        self._reads += 1
        if self.diverge and self._reads > 1:
            images = copy.deepcopy(dict(self.obs.images))
            images["robot0_agentview_left"] = [[[99, 0, 0]]]
            return ObsInput(images=images, proprio=self.obs.proprio, step=self.obs.step)
        return self.obs

    def observation_ref(self):
        return {"step": 0}

    def predicates(self):
        return {"official_check_success": False}


class Manager:
    def __init__(self, env):
        self.env = env
        self.calls = []

    def execute(self, call):
        self.calls.append(call.name)
        self.env.actions += POLICY_CHUNK_ACTIONS
        return SkillResult(
            status=SkillStatus.SUCCESS, skill=call.name, args=call.args,
            instruction=call.instruction, steps=POLICY_CHUNK_ACTIONS,
            success_step=POLICY_CHUNK_ACTIONS, terminated_by="test",
            reason="test", predicates={})


class Trace:
    def __init__(self):
        self.plans = []
        self.results = []
        self.end = None

    def plan(self, *args):
        self.plans.append(args)

    def skill_result(self, result, next_skill=None):
        self.results.append((result, next_skill))

    def episode_end(self, **kwargs):
        self.end = kwargs


class PreEpisodeRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.registry = SkillRegistry(bindings.CONTRACTS)
        self.valid = json.dumps({"plan": [
            {"name": name, "args": dict(self.registry.get(name).default_args)}
            for name in self.registry.names
        ]})

    def run_case(self, output, *, stale=False, diverge=False,
                 status=ResponseStatus.OK):
        env = Environment(diverge=diverge)
        manager = Manager(env)
        trace = Trace()
        service = Service(output, stale=stale, status=status)
        planner = PreEpisodePlanner(self.registry, service, timeout_s=1.0)
        summary = run_preplanned_episode(
            planner, manager, env, trace, self.registry, bindings.GOAL, 600)
        return summary, env, manager, trace, service

    def test_runtime_contract_is_canonical(self):
        self.assertEqual(ACTION_RATE_HZ, 20.0)
        self.assertEqual(POLICY_CHUNK_ACTIONS, 16)
        self.assertEqual(CHUNK_DEADLINE_MS, 800.0)
        self.assertEqual(ACTION_JITTER_DEADLINE_MS, 25.0)
        self.assertEqual(MAX_OBSERVATION_AGE_MS, 1100.0)
        self.assertEqual(MAX_QUEUE_CHUNKS, 1)

    def test_first_action_after_one_complete_plan_and_three_boundaries(self):
        summary, env, manager, trace, service = self.run_case(self.valid)
        self.assertEqual(len(service.requests), 1)
        self.assertEqual(summary["planner_calls"], 1)
        self.assertEqual(summary["runtime_boundary_planner_calls"], 0)
        self.assertEqual(manager.calls, self.registry.names)
        self.assertEqual(len(trace.plans), 3)
        self.assertEqual(env.actions, 48)
        self.assertEqual(summary["steps_used"], 48)

    def test_invalid_timeout_stale_and_divergence_execute_zero_actions(self):
        parsed = json.loads(self.valid)
        cases = {
            "malformed": ("{not-json", {}),
            "unknown": (json.dumps({"plan": [
                {"name": "UNKNOWN", "args": {}}, *parsed["plan"][1:]]}), {}),
            "wrong_sequence": (json.dumps({"plan": [
                parsed["plan"][1], parsed["plan"][0], parsed["plan"][2]]}), {}),
            "extra_field": (json.dumps({"plan": [
                dict(parsed["plan"][0], repair=True), *parsed["plan"][1:]]}), {}),
            "stale": (self.valid, {"stale": True}),
            "divergence": (self.valid, {"diverge": True}),
            "timeout": (TimeoutError("slow planner"), {}),
        }
        for name, (output, kwargs) in cases.items():
            with self.subTest(name=name):
                summary, env, manager, trace, service = self.run_case(output, **kwargs)
                self.assertEqual(len(service.requests), 1)
                self.assertEqual(summary["terminal"], "plan_rejected")
                self.assertEqual(summary["steps_used"], 0)
                self.assertEqual(env.actions, 0)
                self.assertEqual(manager.calls, [])
                self.assertEqual(trace.plans, [])

    def test_service_error_has_no_fallback(self):
        summary, env, manager, _, service = self.run_case(
            self.valid, status=ResponseStatus.ERROR)
        self.assertEqual(len(service.requests), 1)
        self.assertEqual(summary["terminal"], "plan_rejected")
        self.assertEqual(env.actions, 0)
        self.assertEqual(manager.calls, [])


if __name__ == "__main__":
    unittest.main()
