from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from async_policy import validate_policy_lag, validate_trajectory_policy  # noqa: E402


class AsyncPolicyTests(unittest.TestCase):
    def test_current_and_one_version_old_are_accepted(self):
        self.assertEqual(validate_policy_lag(rollout_version=3, learner_version=3), 0)
        self.assertEqual(validate_policy_lag(rollout_version=2, learner_version=3), 1)

    def test_two_versions_old_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "stale rollout"):
            validate_policy_lag(rollout_version=1, learner_version=3)

    def test_future_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "newer than learner"):
            validate_policy_lag(rollout_version=4, learner_version=3)

    def test_episode_cannot_mix_policy_snapshots(self):
        trajectories = {"spark1/traj-1": [
            {"policy_version": 7, "policy_hash": "abc"},
            {"policy_version": 8, "policy_hash": "def"},
        ]}
        with self.assertRaisesRegex(ValueError, "mixes policy snapshots"):
            validate_trajectory_policy(
                trajectories, policy_version=7, policy_hash="abc")


if __name__ == "__main__":
    unittest.main()
