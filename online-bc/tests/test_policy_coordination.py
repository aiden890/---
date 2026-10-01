import json
import tempfile
import unittest
from pathlib import Path

from online_bc.rollout.worker import active_policy_readers, conditional_cup_rate


class PolicyCoordination(unittest.TestCase):
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
