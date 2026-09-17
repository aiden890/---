from pathlib import Path
import unittest


def uses_server_visible_unique_output(script):
    train_case = script.split("  train)\n", 1)[1].split("  g5-sweep)\n", 1)[0]
    return ('--out "/train/results/$run"' in train_case and
            "--out /out" not in train_case)


class RunTrainCheckpointPathTests(unittest.TestCase):
    def test_train_client_uses_server_visible_unique_output_path(self):
        script = (Path(__file__).parents[1] / "scripts" / "run-train.sh").read_text()
        self.assertTrue(uses_server_visible_unique_output(script))

    def test_legacy_shared_output_mutation_is_rejected(self):
        script = (Path(__file__).parents[1] / "scripts" / "run-train.sh").read_text()
        mutated = script.replace('--out "/train/results/$run"', "--out /out", 1)
        self.assertFalse(uses_server_visible_unique_output(mutated))


if __name__ == "__main__":
    unittest.main()