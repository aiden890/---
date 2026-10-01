import unittest
from online_bc.orchestration.adaptation import assess


def report(version, successes, count=10):
    return dict(
        policy_version=version,
        attempts=count,
        cup_successes=successes,
        outcomes=[dict(seed=992001 + i, cup_placed=i < successes) for i in range(count)],
    )


class AdaptationTests(unittest.TestCase):
    def test_common_seeds_prevent_comparing_different_eval_sizes(self):
        base = report(0, 6, 30)
        # Full average is 20%, but the actual common ten have six successes.
        result = assess([base, report(1, 5)])
        self.assertEqual(result["best_version"], 0)
        self.assertEqual(result["best_successes"], 6)

    def test_two_plateau_checkpoints_change_one_variable(self):
        result = assess([report(0, 5), report(1, 5), report(2, 4)])
        self.assertEqual(result["action"], "reduce_learning_rate")
        self.assertEqual(result["learning_rate"], 5e-5)
        self.assertFalse(result["statistically_confirmed"])

    def test_clear_full_eval_regression_preserves_best_resume(self):
        result = assess([report(0, 15, 30), report(1, 6), report(2, 8, 30)])
        self.assertEqual(result["action"], "resume_best_checkpoint")
        self.assertEqual(result["resume_round"], 0)

    def test_improving_checkpoint_continues(self):
        self.assertEqual(assess([report(0, 5), report(1, 6), report(2, 7)])["action"], "continue")


if __name__ == "__main__":
    unittest.main()
