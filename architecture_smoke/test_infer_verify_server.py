from __future__ import annotations

import unittest

from infer_verify_server import InferVerifyServer


class FakeModel:
    def __init__(self):
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return self


class InferVerifyServerTest(unittest.TestCase):
    def setUp(self):
        self.loads = 0

        def load():
            self.loads += 1
            return FakeModel()

        self.server = InferVerifyServer("unused", "127.0.0.1", 0, model_loader=load)
        self.server._base_actions = lambda input_data: (
            "raw-policy-response", input_data["task_id"])
        self.server._op_vlm_score = lambda req: {
            "prob": 0.75, "question": req.get("question")}

    def tearDown(self):
        self.server.close()

    @staticmethod
    def background(**overrides):
        request = {
            "op": "background_vlm",
            "episode_id": "ep-1",
            "skill_id": "GRASP_OBJECT",
            "observation_step": 4,
            "request_id": "req-4",
            "request_kind": "boundary",
            "payload": {"inputs": {"pixels": [1]}, "question": "done?"},
            "timeout_s": 1.0,
        }
        request.update(overrides)
        return request

    def test_model_is_loaded_exactly_once_and_policy_api_stays_raw(self):
        self.assertEqual(self.loads, 1)
        self.assertEqual(self.server.model_load_count, 1)
        self.assertEqual(self.server.model.eval_calls, 1)
        self.assertEqual(
            self.server.handle({"task_id": "robocasa365"}),
            ("raw-policy-response", "robocasa365"),
        )
        self.assertEqual(self.loads, 1)

    def test_canonical_background_envelope_round_trips_identity(self):
        response = self.server.handle(self.background(request_kind="planner"))
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["request_id"], "req-4")
        self.assertEqual(response["request_kind"], "planner")
        self.assertEqual(response["result"]["prob"], 0.75)

    def test_background_rejects_missing_identity_and_simulator_ground_truth(self):
        request = self.background()
        del request["request_id"]
        with self.assertRaisesRegex(ValueError, "missing canonical fields"):
            self.server.handle(request)
        with self.assertRaisesRegex(ValueError, "forbidden field"):
            self.server.handle(self.background(payload={
                "inputs": {}, "official_check_success": True,
            }))

    def test_health_exposes_readiness_queues_counters_and_latency(self):
        self.server.handle(self.background(request_kind="endpoint"))
        health = self.server.handle({"op": "health"})
        self.assertTrue(health["healthy"])
        self.assertTrue(health["ready"])
        self.assertEqual(health["model_load_count"], 1)
        scheduler = health["scheduler"]
        self.assertEqual(scheduler["completed"]["background"], 1)
        self.assertIn("endpoint", scheduler["queue_wait_ms"])
        self.assertIn("endpoint", scheduler["model_latency_ms"])
        for counter in ("dropped", "superseded", "stale", "timeouts"):
            self.assertIn(counter, scheduler)


if __name__ == "__main__":
    unittest.main()
