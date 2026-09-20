#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mibot_rlinf_env import (
    TemporalHistoryBuffer,
    observation_to_mibot_state,
    prepare_mibot_env_actions,
)


class TemporalHistoryBufferTests(unittest.TestCase):
    def test_reset_pads_first_observation(self):
        history = TemporalHistoryBuffer(num_envs=2, length=4, interval=2)
        history.reset("state", 0, 7)
        self.assertEqual(history.values("state", 0), [7, 7, 7, 7])

    def test_stride_matches_legacy_queue_contract(self):
        history = TemporalHistoryBuffer(num_envs=1, length=4, interval=2)
        history.reset("state", 0, 0)
        for value in range(1, 8):
            history.append("state", 0, value)
        self.assertEqual(history.values("state", 0), [1, 3, 5, 7])

    def test_partial_reset_does_not_modify_other_env(self):
        history = TemporalHistoryBuffer(num_envs=2, length=3, interval=1)
        history.reset("image", 0, "a")
        history.reset("image", 1, "b")
        history.append("image", 0, "a1")
        history.append("image", 1, "b1")
        before = history.values("image", 0)
        history.reset("image", 1, "reset")
        self.assertEqual(history.values("image", 0), before)
        self.assertEqual(history.values("image", 1), ["reset"] * 3)

    def test_batch_is_environment_then_time(self):
        history = TemporalHistoryBuffer(num_envs=2, length=2, interval=1)
        history.reset("state", 0, 1)
        history.reset("state", 1, 10)
        history.append("state", 0, 2)
        history.append("state", 1, 20)
        self.assertEqual(history.batch("state", list), [[1, 2], [10, 20]])

    def test_invalid_shape_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            TemporalHistoryBuffer(num_envs=0)
        with self.assertRaises(ValueError):
            TemporalHistoryBuffer(num_envs=1, length=0)
        with self.assertRaises(ValueError):
            TemporalHistoryBuffer(num_envs=1, interval=0)


class MiBoTStateTransformTests(unittest.TestCase):
    def test_rlinf_raw_state_uses_canonical_xiaomi_transform(self):
        import sys
        import types

        captured = {}
        fake_rollout = types.ModuleType("rollout")

        def canonical(obs):
            captured.update(obs)
            return "canonical-14d"

        fake_rollout.observation_to_state = canonical
        old = sys.modules.get("rollout")
        sys.modules["rollout"] = fake_rollout
        try:
            raw = {
                "eef_rel_pos": "p",
                "eef_rel_quat": "q",
                "gripper": "g",
                "base_position": "bp",
                "base_rotation": "bq",
                "irrelevant_velocity": "must-not-enter-state",
            }
            key_map = {
                "base_to_eef_pos": "eef_rel_pos",
                "base_to_eef_quat": "eef_rel_quat",
                "gripper_qpos": "gripper",
                "base_pos": "base_position",
                "base_quat": "base_rotation",
            }
            result = observation_to_mibot_state(raw, key_map)
        finally:
            if old is None:
                sys.modules.pop("rollout", None)
            else:
                sys.modules["rollout"] = old

        self.assertEqual(result, "canonical-14d")
        self.assertEqual(
            captured,
            {
                "state.end_effector_position_relative": "p",
                "state.end_effector_rotation_relative": "q",
                "state.gripper_qpos": "g",
                "state.base_position": "bp",
                "state.base_rotation": "bq",
            },
        )

    def test_action_conversion_matches_xiaomi_gym_wrapper_thresholds(self):
        import numpy as np

        actions = np.zeros((2, 12), dtype=np.float32)
        actions[:, 6] = [0.49, 0.5]
        actions[:, 11] = [0.9, 0.1]
        original = actions.copy()
        converted = prepare_mibot_env_actions(actions)
        np.testing.assert_array_equal(converted[:, 6], [-1.0, 1.0])
        np.testing.assert_array_equal(converted[:, 11], [1.0, -1.0])
        np.testing.assert_array_equal(converted[:, :6], original[:, :6])
        np.testing.assert_array_equal(actions, original)  # caller input is not mutated


if __name__ == "__main__":
    unittest.main()
