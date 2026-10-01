"""Cross-server collection must stay disjoint and respect the success gate."""

import json
import tempfile
import unittest
from pathlib import Path
from online_bc.orchestration.collection import batch_plan, eligible_successes, ready_to_train
from online_bc.data.validate_dataset import validate


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(
            workers={"amp": {}, "v4": {}},
            first_round_attempts=32,
            attempts_per_batch=8,
            max_attempts_per_round=32,
            target_new_successes=8,
        )

    def test_first_collection_is_16_per_host_with_disjoint_seeds(self):
        plan = batch_plan(self.config, 1, 0)
        self.assertEqual([x["episodes"] for x in plan], [16, 16])
        seeds = [
            set(range(x["seed_offset"] + 1, x["seed_offset"] + x["episodes"] + 1)) for x in plan
        ]
        self.assertFalse(seeds[0] & seeds[1])
        self.assertEqual(seeds[0] | seeds[1], set(range(1, 33)))

    def test_later_batches_and_attempt_cap(self):
        self.assertEqual([x["episodes"] for x in batch_plan(self.config, 2, 0)], [4, 4])
        self.assertEqual(sum(x["episodes"] for x in batch_plan(self.config, 2, 30)), 2)
        self.assertEqual(batch_plan(self.config, 2, 32), [])
        self.assertFalse(ready_to_train(self.config, 2, 32, 7))
        self.assertTrue(ready_to_train(self.config, 2, 16, 8))
        self.assertFalse(ready_to_train(self.config, 1, 32, 0))
        self.assertFalse(ready_to_train(self.config, 1, 16, 8))
        self.assertTrue(ready_to_train(self.config, 1, 32, 1))

    def test_success_count_deduplicates_and_respects_review_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            control = Path(directory) / "controls.json"
            control.write_text(
                json.dumps(dict(revision=1, excluded={"pi05-seed1": {}}, paused=False))
            )
            self.assertEqual(
                eligible_successes(
                    [
                        dict(accepted=["pi05-seed1", "pi05-seed2"]),
                        dict(accepted=["pi05-seed2", "pi05-seed3"]),
                    ],
                    control,
                ),
                2,
            )
            self.assertEqual(validate(directory, allow_empty=True)["episodes"], 0)
            with self.assertRaises(AssertionError):
                validate(directory)

    def test_coordinator_queues_one_pi_job_for_two_disjoint_sources(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import sys
        from online_bc.orchestration import online_rounds

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(
                self.config,
                run="test",
                root=str(root / "run"),
                transport_python="fake-python",
                token_file="unused",
                steps_per_round=50,
                learner_models=["pi05"],
                skills=["cup_placement"],
            )
            config["workers"] = {
                node: dict(
                    model="pi05",
                    collect_argv=[
                        "fake",
                        node,
                        "{round}",
                        "{batch}",
                        "{episodes}",
                        "{seed_offset}",
                    ],
                    reload_argv=["reload", node],
                )
                for node in ["amp", "v4"]
            }
            path = root / "config.json"
            path.write_text(json.dumps(config))

            class Process:
                def __init__(self, command, stdout, **kwargs):
                    _, node, version, batch, count, offset = command
                    seeds = list(
                        range(
                            993000 + int(version) * 100 + int(offset) + 1,
                            993000 + int(version) * 100 + int(offset) + int(count) + 1,
                        )
                    )
                    stdout.write(
                        json.dumps(
                            dict(
                                model="pi05",
                                node=node,
                                round=int(version),
                                batch=int(batch),
                                seeds=seeds,
                                accepted=[f"pi05-seed{seeds[0]}"],
                                prefix=f"test/data/pi05/{node}/batch-{batch}",
                                remote_root="unused",
                            )
                        )
                        + "\n"
                    )

                def wait(self):
                    return 0

            def run(command, **kwargs):
                if command[0] == "fake-python" and command[3] == "download":
                    folder = Path(command[4])
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "metadata.json").write_text("{}")
                return SimpleNamespace(returncode=0, stderr="")

            with (
                patch.object(
                    sys, "argv", ["online_rounds", "--config", str(path), "--rounds", "1"]
                ),
                patch.object(online_rounds.subprocess, "Popen", Process),
                patch.object(online_rounds.subprocess, "run", side_effect=run),
            ):
                online_rounds.main()
            job = json.loads((root / "run/jobs/pi05/round-0001/job.json").read_text())
            self.assertEqual(job["attempts"], 32)
            self.assertEqual(job["new_successes"], 2)
            self.assertEqual(len({source["prefix"] for source in job["data_prefixes"]}), 2)
            self.assertEqual({source["node"] for source in job["data_prefixes"]}, {"amp", "v4"})
            self.assertFalse((root / "run/jobs/amp").exists())


if __name__ == "__main__":
    unittest.main()
