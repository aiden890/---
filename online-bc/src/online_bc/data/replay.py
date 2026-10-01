"""Success-gated rolling online BC replay, using RLinf's trajectory cache."""

import json
import random
import hashlib
from pathlib import Path
import numpy as np
import torch
from online_bc._vendor.rlinf_cache import TrajectoryCache
from online_bc.data.data_control import read_controls, select_candidates, episode_id


class Replay:
    def __init__(self, roots, max_episodes=256, seed=42, controls=None):
        self.controls = controls
        self.roots = [Path(p) for p in roots]
        self.capacity = max_episodes
        self.cache = TrajectoryCache(max_size=max_episodes)
        self.episodes = {}
        self.seen = set()
        self.random = random.Random(seed)

    def ingest(self):
        for root in self.roots:
            for path in sorted(root.glob("*/bc/manifest.json")):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in self.seen:
                    continue
                m = json.loads(path.read_text())
                if not m["samples"]:
                    self.seen.add(digest)
                    continue
                tid = len(self.seen)
                self.seen.add(digest)
                actions = []
                valid = []
                for row in m["samples"]:
                    a = np.load(path.parent / row["actions"], allow_pickle=False)
                    assert a["actions"].shape == (50, 12) and a["valid"].shape == (50,)
                    assert np.isfinite(a["actions"]).all() and a["valid"].any()
                    actions.append(torch.from_numpy(a["actions"]))
                    valid.append(torch.from_numpy(a["valid"]))
                self.cache.put(tid, dict(actions=torch.stack(actions), valid=torch.stack(valid)))
                self.episodes[tid] = (path, m)
                while len(self.episodes) > self.capacity:
                    self.episodes.pop(next(iter(self.episodes)))
        return sum(len(m["samples"]) for _, m in self.episodes.values())

    def sample(self, skill=None):
        control = read_controls(self.controls)
        choices = select_candidates(self.episodes, control, skill)
        if not choices:
            raise RuntimeError(
                f"No successful BC samples for skill={skill}. Need teacher successes; never train on failed actions."
            )
        tid, path, m, indices = self.random.choice(choices)
        index = self.random.choice(indices)
        row = m["samples"][index]
        cached = self.cache.get(tid)
        assert cached is not None
        with np.load(path.parent / row["observation"], allow_pickle=False) as obs:
            data = {k: obs[k].copy() for k in obs.files}
        return dict(
            obs=data,
            actions=cached["actions"][index].numpy().copy(),
            valid=cached["valid"][index].numpy().copy(),
            prompt=m["prompt"],
            skill=row["skill"],
            source_model=m["model"],
            seed=m["seed"],
            step=row["step"],
            episode_id=episode_id(m),
            control_revision=control["revision"],
        )
