import unittest

from tracking.make_rlenv_reward_overlays import reward_lines, timeline_row_for_frame


class RewardOverlayTest(unittest.TestCase):
    def test_selects_latest_timeline_row_at_or_before_video_frame(self):
        timeline = [{"f": 0, "s": 1}, {"f": 4, "s": 3}, {"f": 8, "s": 5}]
        self.assertEqual(timeline_row_for_frame(timeline, 6)["s"], 3)

    def test_reward_lines_show_step_delta_components_and_cumulative_total(self):
        row = {
            "s": 120,
            "d": {"term": 1.0, "hold": 0.05, "pen": -0.25},
            "t": 0.8,
        }
        lines = reward_lines(row)
        self.assertIn("step 120", lines[0])
        self.assertIn("delta +0.800", lines[0])
        self.assertIn("cumulative +0.800", lines[0])
        self.assertIn("terminal +1.000", lines[1])
        self.assertIn("hold +0.050", lines[1])
        self.assertIn("penalty -0.250", lines[1])


if __name__ == "__main__":
    unittest.main()
