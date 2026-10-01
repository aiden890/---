import json
import tempfile
import unittest
from pathlib import Path

from online_bc.rollout.worker import (
    active_policy_readers,
    conditional_cup_rate,
    reused_rollouts,
    worker_count,
)


class PolicyCoordination(unittest.TestCase):
    def test_collection_trial_leaves_evaluation_concurrency_unchanged(self):
        config = dict(workers=2, collection_workers=4)
        self.assertEqual(worker_count(config, "collect"), 4)
        self.assertEqual(worker_count(config, "eval"), 2)
        self.assertEqual(worker_count(config, "reload"), 2)
        self.assertEqual(worker_count(dict(workers=2), "collect"), 2)
        with self.assertRaises(ValueError):
            worker_count(dict(workers=2, collection_workers=0), "collect")

    def test_recovery_rejects_a_result_from_another_policy_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "pi05/pi05-seed993441"
            folder.mkdir(parents=True)
            (folder / "result.json").write_text(
                json.dumps(dict(model="pi05", seed=993441, policy_version=3))
            )
            self.assertEqual(reused_rollouts(root, "pi05", [993441, 993442], 3), 1)
            with self.assertRaises(AssertionError):
                reused_rollouts(root, "pi05", [993441], 4)

    def test_conditional_success_excludes_placement_without_grasp(self):
        rows = [dict(grasped=True, cup_placed=True),
                dict(grasped=True, cup_placed=False),
                dict(grasped=False, cup_placed=True)]
        self.assertEqual(conditional_cup_rate(rows), 0.5)
        self.assertIsNone(conditional_cup_rate([rows[-1]]))

    def test_legacy_readers_match_only_the_same_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "worker.json"
            config.write_text(json.dumps({}))
            for pid, action, filename in [
                (70001, "eval", config),
                (70002, "collect", config),
                (70003, "reload", config),
                (70004, "eval", root / "other.json"),
            ]:
                folder = root / str(pid)
                folder.mkdir()
                argv = ["python3", "-m", "online_bc.rollout.worker", "--config", str(filename), action]
                (folder / "cmdline").write_bytes(("\0".join(argv) + "\0").encode())
            self.assertEqual(sorted(active_policy_readers(config, root)), [70001, 70002])

    def test_disappeared_process_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "70001").mkdir()
            self.assertEqual(active_policy_readers(root / "worker.json", root), [])

    def test_reader_waiting_on_lock_does_not_deadlock_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "worker.json"
            lock = root / "policy.lock"
            lock.touch()
            folder = root / "70001"
            folder.mkdir()
            argv = ["python3", "-m", "online_bc.rollout.worker", "--config", str(config), "eval"]
            (folder / "cmdline").write_bytes(("\0".join(argv) + "\0").encode())
            (folder / "fd").mkdir()
            (folder / "fd" / "3").symlink_to(lock)
            self.assertEqual(active_policy_readers(config, root, lock), [])


if __name__ == "__main__":
    unittest.main()
