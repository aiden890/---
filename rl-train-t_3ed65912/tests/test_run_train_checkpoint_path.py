from pathlib import Path
import unittest


class RunTrainCheckpointPathTests(unittest.TestCase):
    def test_train_client_uses_server_visible_unique_output_path(self):
        script = (Path(__file__).parents[1] / "scripts" / "run-train.sh").read_text()
        train_case = script.split("  train)\n", 1)[1].split("  g5-sweep)\n", 1)[0]

        self.assertIn('--out "/train/results/$run"', train_case)
        self.assertNotIn("--out /out", train_case)


if __name__ == "__main__":
    unittest.main()