import unittest
from online_bc.orchestration.collection import batch_plan


class ExtraWindowTests(unittest.TestCase):
    def test_extra_namespace_preserves_round_stride_and_exhausts_at_192(self):
        c = dict(
            workers={"amp": {}, "v4": {}},
            max_attempts_per_round=96,
            attempts_per_batch=8,
            additional_windows={"12": [dict(attempt_start=96, attempts=96, seed_start=2012001)]},
        )
        used = set()
        for attempted in range(96, 192, 8):
            for p in batch_plan(c, 12, attempted):
                seeds = set(
                    range(
                        994200 + p["seed_offset"] + 1, 994200 + p["seed_offset"] + p["episodes"] + 1
                    )
                )
                self.assertFalse(seeds & used)
                used |= seeds
        self.assertEqual(used, set(range(2012001, 2012097)))
        self.assertFalse(used & set(range(992001, 992031)))
        self.assertEqual(batch_plan(c, 12, 192), [])
        self.assertEqual(batch_plan(c, 13, 96), [])

    def test_native_window_remains_unchanged(self):
        c = dict(
            workers={"amp": {}, "v4": {}},
            max_attempts_per_round=96,
            attempts_per_batch=8,
            additional_windows={"12": [dict(attempt_start=96, attempts=96, seed_start=2012001)]},
        )
        self.assertEqual([p["seed_offset"] for p in batch_plan(c, 12, 88)], [88, 92])


if __name__ == "__main__":
    unittest.main()


class PublicMappingTests(unittest.TestCase):
    def test_action_mapping_preserves_all_commands(self):
        import numpy as np
        from online_bc.data.build_public_grasp import native_actions, native_state

        a = np.arange(24).reshape(2, 12)
        b = native_actions(a)
        np.testing.assert_array_equal(b, a[:, [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]])
        s = native_state(np.arange(16))
        np.testing.assert_array_equal(s["state.end_effector_rotation_relative"], [10, 11, 12, 13])
        np.testing.assert_array_equal(s["state.base_rotation"], [3, 4, 5, 6])

    def test_expert_pool_does_not_evict_native_and_honors_exclusions(self):
        import tempfile
        import json
        import numpy as np
        from pathlib import Path
        from online_bc.data.replay import Replay
        from online_bc.data.data_control import select_candidates

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index in range(6):
                f = root / f"episode-{index}" / "bc"
                f.mkdir(parents=True)
                np.savez(f / "obs.npz", state=np.array([index]))
                np.savez(
                    f / "actions.npz",
                    actions=np.zeros((50, 12), np.float32),
                    valid=np.ones(50, bool),
                )
                expert = index < 3
                (f / "manifest.json").write_text(
                    json.dumps(
                        dict(
                            model="expert-human" if expert else "pi05",
                            seed=index,
                            episode_id=f"expert-{index}" if expert else f"pi05-seed{index}",
                            dataset_role="expert_grasp_base" if expert else "online",
                            prompt="test",
                            samples=[
                                dict(
                                    skill="grasp" if expert else "cup_placement",
                                    observation="obs.npz",
                                    actions="actions.npz",
                                    step=0,
                                )
                            ],
                        )
                    )
                )
            replay = Replay([root], max_episodes=2)
            replay.ingest()
            self.assertEqual(len(replay.expert_cache), 3)
            self.assertEqual(len(replay.episodes), 5)
            choices = select_candidates(
                replay.episodes, dict(excluded={"expert-0": {}}, revision=1, paused=False), "grasp"
            )
            self.assertEqual({m["episode_id"] for _, _, m, _ in choices}, {"expert-1", "expert-2"})
            self.assertEqual(replay.sample("grasp")["skill"], "grasp")
            self.assertEqual(replay.sample("cup_placement")["skill"], "cup_placement")
