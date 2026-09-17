from __future__ import annotations

import threading
import tempfile
import unittest
from pathlib import Path

from async_control import (
    AsyncVerifierClient, ControlRequest, ControlResponse, RequestKind, ResponseStatus,
)


def request(step: int, *, episode: str = "ep", skill: str = "GRASP_OBJECT",
            kind: RequestKind = RequestKind.BOUNDARY, request_id: str | None = None,
            payload=None) -> ControlRequest:
    return ControlRequest(episode, skill, step, request_id or f"r{step}", kind,
                          payload or {"images": {"eye": [[step]]}, "proprio": [0.0]})


def ok(req: ControlRequest, recommendation: str = "CONTINUE") -> ControlResponse:
    return ControlResponse.from_request(req, recommendation=recommendation)


class AsyncControlTest(unittest.TestCase):
    def test_latest_pending_atomically_supersedes_without_preempting_inflight(self):
        first_started = threading.Event()
        release_first = threading.Event()
        calls = []

        def transport(req, timeout):
            calls.append(req.request_id)
            if req.request_id == "r1":
                first_started.set()
                self.assertTrue(release_first.wait(timeout))
            return ok(req)

        events = []
        client = AsyncVerifierClient(
            transport, timeout_s=1,
            telemetry=lambda event, **kw: events.append((event, kw)))
        client.set_context("ep", "GRASP_OBJECT")
        client.submit(request(1))
        self.assertTrue(first_started.wait(1))
        self.assertIsNone(client.submit(request(2)))
        replaced = client.submit(request(3))
        self.assertIsNotNone(replaced)
        self.assertEqual(replaced.request_id, "r2")
        self.assertEqual(client.in_flight.request_id, "r1")
        release_first.set()
        self.assertTrue(client.wait_idle(1))
        responses = client.poll()
        client.close(timeout=1)

        self.assertEqual(calls, ["r1", "r3"])
        self.assertEqual([r.request_id for r in responses], ["r1", "r3"])
        self.assertTrue(any(event == "drop" and data["reason"] == "superseded"
                            for event, data in events))

    def test_never_runs_more_than_one_rpc_inflight(self):
        release = threading.Event()
        entered = threading.Event()
        lock = threading.Lock()
        active = 0
        peak = 0

        def transport(req, timeout):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            entered.set()
            self.assertTrue(release.wait(timeout))
            with lock:
                active -= 1
            return ok(req)

        client = AsyncVerifierClient(transport, timeout_s=1)
        client.submit(request(1))
        self.assertTrue(entered.wait(1))
        client.submit(request(2))
        client.submit(request(3))
        release.set()
        self.assertTrue(client.wait_idle(1))
        client.close(timeout=1)
        self.assertEqual(peak, 1)

    def test_stale_episode_skill_and_step_are_dropped_with_reasons(self):
        cases = [
            (request(2, episode="old"), ("ep", "GRASP_OBJECT", -1), "stale_episode"),
            (request(2, skill="MOVE_OBJECT"), ("ep", "GRASP_OBJECT", -1), "stale_skill"),
            (request(2), ("ep", "GRASP_OBJECT", 3), "stale_step"),
        ]
        for response_request, context, reason in cases:
            with self.subTest(reason=reason):
                events = []
                client = AsyncVerifierClient(
                    lambda req, timeout: ok(req),
                    telemetry=lambda event, **kw: events.append((event, kw)))
                client.set_context(*context)
                client.submit(response_request)
                self.assertTrue(client.wait_idle(1))
                self.assertEqual(client.poll(), [])
                client.close(timeout=1)
                self.assertTrue(any(event == "drop" and data["reason"] == reason
                                    for event, data in events))

    def test_timeout_retries_once_with_same_identity_then_reports_timeout(self):
        identities = []

        def transport(req, timeout):
            identities.append(req.identity)
            raise TimeoutError("rpc deadline")

        client = AsyncVerifierClient(transport, timeout_s=0.01)
        req = request(4, request_id="stable-id")
        client.set_context(req.episode_id, req.skill_id)
        client.submit(req)
        self.assertTrue(client.wait_idle(1))
        responses = client.poll()
        client.close(timeout=1)

        self.assertEqual(identities, [req.identity, req.identity])
        self.assertEqual(len(responses), 1)
        self.assertIs(responses[0].status, ResponseStatus.TIMEOUT)
        self.assertEqual(responses[0].attempt, 1)

    def test_shutdown_cancels_pending_and_joins_worker_without_leak(self):
        started = threading.Event()
        release = threading.Event()
        calls = []

        def transport(req, timeout):
            calls.append(req.request_id)
            started.set()
            self.assertTrue(release.wait(timeout))
            return ok(req)

        client = AsyncVerifierClient(transport, timeout_s=1)
        client.submit(request(1))
        self.assertTrue(started.wait(1))
        client.submit(request(2))
        release.set()
        client.close(drain=False, timeout=1)
        self.assertEqual(calls, ["r1"])
        self.assertFalse(client.is_alive)

    def test_request_kinds_round_trip_and_remote_response_is_only_advisory(self):
        for kind in RequestKind:
            req = request(1, kind=kind, request_id=kind.value)
            response = ok(req, recommendation="ADVANCE")
            self.assertEqual(response.identity, req.identity)
            self.assertEqual(response.recommendation, "ADVANCE")
            self.assertFalse(hasattr(response, "apply"))

    def test_runtime_request_rejects_forbidden_ground_truth_recursively(self):
        with self.assertRaisesRegex(ValueError, "forbidden field"):
            request(1, payload={"observation": {"official_check_success": True}})
        with self.assertRaisesRegex(ValueError, "forbidden field"):
            request(1, payload={"planner": [{"ground_truth": "success"}]})

    def test_execution_manager_keeps_stepping_while_verifier_is_inflight(self):
        import bindings
        from executor import ExecutionManager
        from obs_verifier import ObsInput
        from policy import MockPolicy
        from schemas import AdapterMode, SkillCall
        from skills import SkillRegistry
        from trace import Trace

        started = threading.Event()
        release = threading.Event()
        completed_normally = threading.Event()

        class BlockingBackend:
            def score_view(self, images, question, view="full"):
                started.set()
                if not release.wait(1):
                    raise AssertionError("control loop blocked on verifier RPC")
                completed_normally.set()
                return 0.1

        class Env:
            CAMS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")

            def __init__(self):
                self.steps = 0

            def build_policy_input(self, instruction, adapter_mode, adapter_checkpoint=None):
                from schemas import PolicyInput
                return PolicyInput(instruction, [], {}, adapter_mode, adapter_checkpoint)

            def step(self, action):
                self.steps += 1
                if self.steps == 2:
                    if not started.wait(1):
                        raise AssertionError("async verifier never started")
                    release.set()
                return {}, self.steps == 2, False, {}

            def obs_for_verifier(self):
                image = [[[0, 0, 0]]]
                return ObsInput({key: image for key in self.CAMS}, [0.0] * 14, self.steps)

            def maybe_record_frame(self, force=False):
                return None

            def predicates(self):
                raise AssertionError("privileged predicates entered async runtime")

        tmp = Path(tempfile.mkdtemp())
        trace = Trace(tmp / "trace.jsonl", {"episode_id": "manager-test"})
        env = Env()
        manager = ExecutionManager(
            SkillRegistry(bindings.CONTRACTS), MockPolicy(replan_steps=2, action_dim=12),
            env, trace, AdapterMode.DISABLED, vlm_backend=BlockingBackend(),
            vlm_min_interval=1, hysteresis_k=1, event_gated=False,
            episode_id="manager-test", verifier_timeout_s=1)
        call = SkillCall("MOVE_OBJECT", bindings.CONTRACTS[1].default_args)
        result = manager.execute(call)
        trace.close()

        self.assertEqual(env.steps, 2)
        self.assertTrue(completed_normally.is_set())
        self.assertEqual(result.terminated_by, "env_terminated")
        records = (tmp / "trace.jsonl").read_text().splitlines()
        self.assertTrue(any('"type": "async_control"' in line for line in records))


if __name__ == "__main__":
    unittest.main()
