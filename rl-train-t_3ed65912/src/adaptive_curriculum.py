"""Online moving-band seed curriculum with exact resumability."""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path


class AdaptiveCurriculum:
    """Track current-policy seed difficulty and sample the moving middle band."""

    def __init__(self, seed_base, universe, band, group, explore_frac=0.25,
                 ema=0.5, rng_seed=12345, avoid_recent=3):
        self.seed_base = int(seed_base)
        self.universe = [self.seed_base + i for i in range(int(universe))]
        self.lo, self.hi = int(band[0]), int(band[1])
        self.group = int(group)
        self.explore_frac = float(explore_frac)
        self.ema = float(ema)
        self.avoid_recent = int(avoid_recent)
        self.rng = random.Random(int(rng_seed))
        self.difficulty = {}
        self.recent = []
        self.history = []

    def to_dict(self):
        return {
            "seed_base": self.seed_base,
            "universe": len(self.universe),
            "band": [self.lo, self.hi],
            "group": self.group,
            "explore_frac": self.explore_frac,
            "ema": self.ema,
            "avoid_recent": self.avoid_recent,
            "difficulty": {str(k): copy.deepcopy(v) for k, v in self.difficulty.items()},
            "recent": list(self.recent),
            "rng_state": json.dumps(self.rng.getstate()),
            "history": copy.deepcopy(self.history),
        }

    def load_state(self, state):
        self.difficulty = {int(k): v for k, v in state.get("difficulty", {}).items()}
        self.history = list(state.get("history", []))
        self.recent = [int(seed) for seed in state.get("recent", [])]
        rng_state = state.get("rng_state")
        if rng_state is not None:
            decoded = json.loads(rng_state)
            self.rng.setstate((decoded[0], tuple(decoded[1]), decoded[2]))
            return
        # Compatibility only for caches created before exact RNG state was persisted.
        for _ in self.history:
            self.rng.random()

    def _in_band(self):
        return [seed for seed, value in self.difficulty.items()
                if self.lo <= value.get("ema", -1) <= self.hi]

    def _unseen(self):
        return [seed for seed in self.universe if seed not in self.difficulty]

    def next_seed(self, it, exclude=()):
        """Return a seed/source pair, excluding seeds already used in this update."""
        del it  # retained in the public API for audit call sites
        excluded = {int(seed) for seed in exclude}
        in_band = [seed for seed in self._in_band() if seed not in excluded]
        unseen = [seed for seed in self._unseen() if seed not in excluded]
        available = [seed for seed in self.universe if seed not in excluded]
        if not available:
            raise RuntimeError("adaptive seed universe exhausted by distinct-seed requirement")

        explore = (self.rng.random() < self.explore_frac) or not in_band
        source = "explore"
        if explore and unseen:
            seed = self.rng.choice(unseen)
        elif in_band:
            recent = set(self.recent[-self.avoid_recent:]) if self.avoid_recent else set()
            pool = [seed for seed in in_band if seed not in recent] or in_band
            seed = self.rng.choice(pool)
            source = "exploit"
        elif unseen:
            seed = self.rng.choice(unseen)
        else:
            centre = (self.lo + self.hi) / 2.0
            seed = min(available, key=lambda item: abs(self.difficulty[item]["ema"] - centre))
            source = "nearest"
        if not self.recent or self.recent[-1] != seed:
            self.recent.append(seed)
        return int(seed), source

    def update(self, seed, n_succ, it=None, source=None):
        seed = int(seed)
        n_succ = int(n_succ)
        current = self.difficulty.get(seed)
        if current is None:
            self.difficulty[seed] = {"ema": float(n_succ), "count": 1, "last_n": n_succ}
        else:
            current["ema"] = self.ema * n_succ + (1.0 - self.ema) * current["ema"]
            current["count"] += 1
            current["last_n"] = n_succ
        self.history.append({"iter": it, "seed": seed, "n_succ": n_succ, "source": source})

    def band_stats(self):
        below = sum(1 for value in self.difficulty.values() if value["ema"] < self.lo)
        in_band = sum(1 for value in self.difficulty.values()
                      if self.lo <= value["ema"] <= self.hi)
        above = sum(1 for value in self.difficulty.values() if value["ema"] > self.hi)
        return {"seen": len(self.difficulty), "below": below,
                "in_band": in_band, "above": above}


def atomic_write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


class CurriculumTransaction:
    """Rollback adaptive draws on failed batches; commit cache atomically on success."""

    def __init__(self, curriculum, cache_path=None):
        self.curriculum = curriculum
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.snapshot = curriculum.to_dict()
        self.committed = False

    def __enter__(self):
        return self

    def commit(self):
        if self.cache_path is not None:
            atomic_write_json(self.cache_path, self.curriculum.to_dict())
        self.committed = True

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None or not self.committed:
            self.curriculum.load_state(self.snapshot)
            if self.cache_path is not None:
                temporary = self.cache_path.with_name(self.cache_path.name + ".tmp")
                temporary.unlink(missing_ok=True)
        return False
