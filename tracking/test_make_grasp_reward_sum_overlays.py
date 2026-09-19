import json
import unittest
from pathlib import Path

from tracking.make_grasp_reward_sum_overlays import (
    build_ffmpeg_filter,
    build_record,
    reward_transition_frame,
)


class GraspRewardSumOverlayTest(unittest.TestCase):
    def test_reward_transition_frame_matches_saved_video_schedule(self):
        self.assertEqual(reward_transition_frame(129, video_stride=2), 65)
        self.assertEqual(reward_transition_frame(156, video_stride=2), 78)

    def test_success_record_exposes_exact_decayed_reward_and_transition_time(self):
        row = {
            "job_id": "cfg-abc-ep00001",
            "seed": 710001,
            "reward": 0.9821433300119734,
            "success": True,
            "steps": 129,
            "hold_stats": {"success_step": 129},
        }
        record = build_record(row, "raw.mp4", "overlay.mp4", video_stride=2, video_fps=20)
        self.assertEqual(record["reward_sum"], 0.9821433300119734)
        self.assertEqual(record["reward_frame"], 65)
        self.assertEqual(record["reward_time_seconds"], 3.25)
        self.assertEqual(record["outcome"], "SUCCESS")

    def test_failure_record_keeps_zero_sum_without_reward_transition(self):
        row = {
            "job_id": "cfg-abc-ep00000",
            "seed": 710000,
            "reward": 0.0,
            "success": False,
            "steps": 208,
            "hold_stats": {"success_step": None},
        }
        record = build_record(row, "raw.mp4", "overlay.mp4", video_stride=2, video_fps=20)
        self.assertEqual(record["reward_sum"], 0.0)
        self.assertIsNone(record["reward_frame"])
        self.assertIsNone(record["reward_time_seconds"])
        self.assertEqual(record["outcome"], "TIMEOUT")

    def test_ffmpeg_filter_switches_from_zero_to_exact_reward_at_success_frame(self):
        record = {
            "seed": 710001,
            "outcome": "SUCCESS",
            "success_step": 129,
            "reward_sum": 0.9821433300119734,
            "reward_frame": 65,
        }
        filter_spec = build_ffmpeg_filter(record)
        self.assertIn("누적 리워드 합계 0.000000", filter_spec)
        self.assertIn("누적 리워드 합계 0.982143", filter_spec)
        self.assertIn("lt(n\\,65)", filter_spec)
        self.assertIn("gte(n\\,65)", filter_spec)

    def test_tracking_site_exposes_exactly_ten_reward_rollouts(self):
        root = Path(__file__).resolve().parent
        manifest = json.loads((root / "grasp_reward_rollouts.json").read_text())
        self.assertEqual(len(manifest["rollouts"]), 10)
        self.assertEqual(len({row["seed"] for row in manifest["rollouts"]}), 10)
        self.assertTrue(all((root / row["mp4"]).is_file() for row in manifest["rollouts"]))

        page = root / "experiments/exp-20260919-grasp-reward-rollouts.html"
        self.assertTrue(page.is_file())
        self.assertIn("grasp_reward_rollouts.json", page.read_text())
        self.assertIn("experiments/exp-20260919-grasp-reward-rollouts.html", (root / "index.html").read_text())


if __name__ == "__main__":
    unittest.main()
