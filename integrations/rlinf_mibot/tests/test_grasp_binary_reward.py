from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grasp_binary_reward import StableGraspRewardEnv, StableGraspTracker


class _FakeCounter:
    pass


class _FakeEnv:
    def __init__(self):
        self.z = 0.30
        self.contact = False
        self.on_counter = True
        self.lid = types.SimpleNamespace(name="lid")
        self.blender = types.SimpleNamespace(blender_lid=self.lid)
        self.gripper = object()
        self.robots = [types.SimpleNamespace(gripper={"right": self.gripper})]
        self.counter = _FakeCounter()
        self.fixtures = {"counter": self.counter}
        data = types.SimpleNamespace(get_body_xpos=lambda _name: [0.0, 0.0, self.z])
        self.sim = types.SimpleNamespace(data=data)

    def reset(self, **_kwargs):
        self.z = 0.30
        self.contact = False
        self.on_counter = True
        return {"observation": 1}

    def step(self, action):
        return {"observation": action}, 0.0, False, {}

    def check_contact(self, first, second):
        if first is self.gripper and second is self.lid:
            return self.contact
        if first is self.lid and second is self.counter:
            return self.on_counter
        return False


def _fake_robocasa_modules():
    fixtures = types.ModuleType("robocasa.models.fixtures")
    fixtures.Counter = _FakeCounter
    models = types.ModuleType("robocasa.models")
    models.fixtures = fixtures
    robocasa = types.ModuleType("robocasa")
    robocasa.models = models
    return {
        "robocasa": robocasa,
        "robocasa.models": models,
        "robocasa.models.fixtures": fixtures,
    }


class StableGraspRewardTests(unittest.TestCase):
    def test_success_requires_consecutive_steps_and_latches(self):
        tracker = StableGraspTracker(hold_steps=3)
        self.assertFalse(tracker.update(True))
        self.assertFalse(tracker.update(True))
        self.assertFalse(tracker.update(False))
        self.assertFalse(tracker.update(True))
        self.assertFalse(tracker.update(True))
        self.assertTrue(tracker.update(True))
        self.assertTrue(tracker.update(False))

    def test_reset_clears_streak_and_latch(self):
        tracker = StableGraspTracker(hold_steps=2)
        tracker.update(True)
        self.assertTrue(tracker.update(True))
        tracker.reset()
        self.assertFalse(tracker.success)
        self.assertEqual(tracker.streak, 0)

    def test_contact_lift_and_counter_clearance_are_all_required(self):
        with patch.dict(sys.modules, _fake_robocasa_modules()):
            raw = _FakeEnv()
            env = StableGraspRewardEnv(raw, hold_steps=3, lift_dz=0.05)
            self.assertEqual(env.reset(), {"observation": 1})
            raw.contact = True
            raw.z = 0.35
            raw.on_counter = False
            self.assertFalse(env._check_success())
            raw.z = 0.36
            raw.on_counter = True
            self.assertFalse(env._check_success())
            raw.on_counter = False
            self.assertFalse(env._check_success())
            self.assertFalse(env._check_success())
            self.assertTrue(env._check_success())

    def test_relative_chunk_reward_is_exactly_one(self):
        with patch.dict(sys.modules, _fake_robocasa_modules()):
            raw = _FakeEnv()
            env = StableGraspRewardEnv(raw, hold_steps=3, lift_dz=0.05)
            env.reset()
            raw.contact = True
            raw.on_counter = False
            raw.z = 0.36
            successes = [env._check_success() for _ in range(6)]
            previous = False
            relative = []
            for success in successes:
                relative.append(float(success) - float(previous))
                previous = success
            self.assertEqual(successes, [False, False, True, True, True, True])
            self.assertEqual(relative, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            self.assertEqual(sum(relative), 1.0)


if __name__ == "__main__":
    unittest.main()
