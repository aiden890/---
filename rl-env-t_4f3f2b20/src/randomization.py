"""Domain randomization for CloseBlenderLid skill-conditioned RL.

Design goals (from task t_4f3f2b20):
  * Every randomized quantity is drawn from an explicit, seeded distribution so a
    rollout is exactly reproducible from (split, episode_index) alone.
  * train / validation / test variation ranges are *disjoint* where the task asks
    for held-out generalization (asset variants, lid-handle geometry, scene seeds),
    and shared-but-resampled where only pose/physics noise differs.
  * Physically implausible combinations are rejected by validation rules and
    resampled, so the sampler never emits a scene that cannot be built.
  * The full drawn parameter vector is serialized per rollout (`to_metadata`) and can
    be replayed (`RandomizationSpec.from_metadata`) to rebuild the identical scene.

This module is deliberately simulator-agnostic: it produces a plain dict of numbers
and asset choices. The rollout driver applies them to the RoboCasa env (initial lid
pose/yaw, blender pose, robot base/eef, camera, lighting, physics) at reset time.
Applying a field to MuJoCo is the driver's job; here we only *decide and record*.

Ranges are intentionally conservative for the RTX-3090 pilot and are all overridable
from the JSON config (configs/randomization.json) so the multi-GPU config can widen
them without code changes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

SPLITS = ("train", "validation", "test")

# ---------------------------------------------------------------------------
# Default randomization ranges. Everything here is data; the config file can
# override any leaf. Poses are metres / radians; physics are multiplicative.
# ---------------------------------------------------------------------------
DEFAULT_RANGES: dict[str, Any] = {
    "lid_pose": {
        "xy_offset_m": [-0.04, 0.04],   # planar jitter of the lid start pose
        "z_offset_m": [0.0, 0.03],      # lifted slightly above rest at most
        "yaw_rad": [-0.5236, 0.5236],   # +/- 30 deg
    },
    "blender_pose": {
        "xy_offset_m": [-0.03, 0.03],
        "yaw_rad": [-0.2618, 0.2618],   # +/- 15 deg
    },
    "robot_base": {
        "xy_offset_m": [-0.05, 0.05],
        "yaw_rad": [-0.1745, 0.1745],   # +/- 10 deg
    },
    "robot_eef": {
        "xyz_offset_m": [-0.02, 0.02],
    },
    "camera": {
        "pos_noise_m": [-0.01, 0.01],
        "rot_noise_rad": [-0.0175, 0.0175],   # +/- 1 deg calibration noise
        "fov_noise_deg": [-1.0, 1.0],
    },
    "lighting": {
        "intensity_scale": [0.8, 1.2],
        "ambient_scale": [0.7, 1.3],
    },
    "physics": {
        "lid_mass_scale": [0.85, 1.15],
        "friction_scale": [0.8, 1.2],
        "contact_damping_scale": [0.8, 1.25],
    },
    "clutter": {
        "num_distractors": [0, 2],     # inclusive integer range
    },
}

# Asset variant pools. The lid-handle geometry / lid asset is the held-out axis:
# train and test use *disjoint* pools so "test" measures generalization to unseen
# lid handle geometry, exactly as the task requires. validation overlaps train's
# distribution family but uses its own seed stream.
DEFAULT_ASSET_POOLS: dict[str, Any] = {
    # Real RoboCasa Blender lid asset ids observed in this project (Blender028 etc).
    # Kept as opaque ids; the driver maps id -> fixtures/blenders/<id>/... xml.
    "lid_variants": {
        "train": ["Blender001", "Blender003", "Blender010", "Blender019", "Blender028"],
        "validation": ["Blender001", "Blender010", "Blender028"],
        "test": ["Blender005", "Blender023", "Blender041"],
    },
    # Scene layout/style seeds are split disjointly too.
    "scene_seed_pools": {
        "train": {"seed_lo": 0, "seed_hi": 799},
        "validation": {"seed_lo": 800, "seed_hi": 899},
        "test": {"seed_lo": 900, "seed_hi": 999},
    },
}


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


@dataclass
class RandomizationSpec:
    """A fully-resolved, reproducible set of scene parameters for one rollout."""

    split: str
    episode_index: int
    scene_seed: int
    lid_variant: str
    values: dict[str, Any]           # the drawn numeric fields, grouped
    ranges_digest: str               # sha256 of the ranges used (provenance)
    asset_digest: str                # sha256 of the asset pools used
    valid: bool = True
    rejections: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_metadata(meta: dict[str, Any]) -> "RandomizationSpec":
        return RandomizationSpec(**meta)


class Randomizer:
    """Seeded sampler that turns (split, episode_index) into a RandomizationSpec.

    Reproducibility: the RNG is seeded from a stable hash of
    (master_seed, split, episode_index), so the same triple always yields the same
    scene regardless of call order or machine.
    """

    def __init__(
        self,
        master_seed: int = 0,
        ranges: Optional[dict] = None,
        asset_pools: Optional[dict] = None,
    ) -> None:
        self.master_seed = int(master_seed)
        self.ranges = json.loads(json.dumps(ranges)) if ranges else json.loads(json.dumps(DEFAULT_RANGES))
        self.asset_pools = json.loads(json.dumps(asset_pools)) if asset_pools else json.loads(json.dumps(DEFAULT_ASSET_POOLS))
        self._ranges_digest = _digest(self.ranges)
        self._asset_digest = _digest(self.asset_pools)

    def _rng(self, split: str, episode_index: int) -> np.random.Generator:
        key = f"{self.master_seed}|{split}|{episode_index}".encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        return np.random.default_rng(seed)

    def sample(self, split: str, episode_index: int, max_resample: int = 16) -> RandomizationSpec:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        rng = self._rng(split, episode_index)
        rejections: list[str] = []

        for _ in range(max_resample):
            spec = self._draw(rng, split, episode_index)
            problems = validate_scene(spec, self.ranges)
            if not problems:
                spec.valid = True
                spec.rejections = rejections
                return spec
            rejections.extend(problems)
        # Exhausted resamples: return the last draw flagged invalid so the caller
        # can decide (skip the episode) rather than silently building a bad scene.
        spec.valid = False
        spec.rejections = rejections
        return spec

    def _draw(self, rng: np.random.Generator, split: str, episode_index: int) -> dict:
        R = self.ranges
        pools = self.asset_pools

        lid_pool = pools["lid_variants"][split]
        lid_variant = lid_pool[int(rng.integers(0, len(lid_pool)))]

        sp = pools["scene_seed_pools"][split]
        scene_seed = int(rng.integers(sp["seed_lo"], sp["seed_hi"] + 1))

        clutter_lo, clutter_hi = R["clutter"]["num_distractors"]
        values = {
            "lid_pose": {
                "xy_offset_m": [_u(rng, *R["lid_pose"]["xy_offset_m"]), _u(rng, *R["lid_pose"]["xy_offset_m"])],
                "z_offset_m": _u(rng, *R["lid_pose"]["z_offset_m"]),
                "yaw_rad": _u(rng, *R["lid_pose"]["yaw_rad"]),
            },
            "blender_pose": {
                "xy_offset_m": [_u(rng, *R["blender_pose"]["xy_offset_m"]), _u(rng, *R["blender_pose"]["xy_offset_m"])],
                "yaw_rad": _u(rng, *R["blender_pose"]["yaw_rad"]),
            },
            "robot_base": {
                "xy_offset_m": [_u(rng, *R["robot_base"]["xy_offset_m"]), _u(rng, *R["robot_base"]["xy_offset_m"])],
                "yaw_rad": _u(rng, *R["robot_base"]["yaw_rad"]),
            },
            "robot_eef": {
                "xyz_offset_m": [_u(rng, *R["robot_eef"]["xyz_offset_m"]) for _ in range(3)],
            },
            "camera": {
                "pos_noise_m": [_u(rng, *R["camera"]["pos_noise_m"]) for _ in range(3)],
                "rot_noise_rad": [_u(rng, *R["camera"]["rot_noise_rad"]) for _ in range(3)],
                "fov_noise_deg": _u(rng, *R["camera"]["fov_noise_deg"]),
            },
            "lighting": {
                "intensity_scale": _u(rng, *R["lighting"]["intensity_scale"]),
                "ambient_scale": _u(rng, *R["lighting"]["ambient_scale"]),
            },
            "physics": {
                "lid_mass_scale": _u(rng, *R["physics"]["lid_mass_scale"]),
                "friction_scale": _u(rng, *R["physics"]["friction_scale"]),
                "contact_damping_scale": _u(rng, *R["physics"]["contact_damping_scale"]),
            },
            "clutter": {
                "num_distractors": int(rng.integers(clutter_lo, clutter_hi + 1)),
            },
        }
        return RandomizationSpec(
            split=split,
            episode_index=episode_index,
            scene_seed=scene_seed,
            lid_variant=lid_variant,
            values=values,
            ranges_digest=self._ranges_digest,
            asset_digest=self._asset_digest,
        )


def validate_scene(spec: RandomizationSpec, ranges: dict) -> list[str]:
    """Physical-plausibility rules. Returns a list of violated-rule names (empty = ok).

    These reject combinations that would produce an unbuildable or nonsensical scene.
    They are intentionally simple, explicit, and unit-tested.
    """
    problems: list[str] = []
    v = spec.values

    # 1. Lid must not start intersecting the counter (large negative z with big xy jump).
    z = v["lid_pose"]["z_offset_m"]
    if z < -1e-6:
        problems.append("lid_z_below_rest")

    # 2. Physics scales must stay positive and within a stable contact regime.
    ph = v["physics"]
    for name in ("lid_mass_scale", "friction_scale", "contact_damping_scale"):
        if ph[name] <= 0:
            problems.append(f"nonpositive_{name}")
    # Very low mass with very high friction is a known MuJoCo contact-instability combo.
    if ph["lid_mass_scale"] < 0.9 and ph["friction_scale"] > 1.15:
        problems.append("low_mass_high_friction_unstable")

    # 3. Combined blender + robot-base displacement must not exceed reachable workspace.
    b = np.linalg.norm(v["blender_pose"]["xy_offset_m"])
    rb = np.linalg.norm(v["robot_base"]["xy_offset_m"])
    if b + rb > 0.12:
        problems.append("blender_base_out_of_reach")

    # 4. Lighting must remain in a range where cameras are not fully black/blown out.
    li = v["lighting"]
    if li["intensity_scale"] <= 0.0 or li["ambient_scale"] <= 0.0:
        problems.append("nonpositive_lighting")

    return problems


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def splits_disjoint(asset_pools: Optional[dict] = None) -> dict[str, bool]:
    """Report which held-out axes are truly disjoint between train and test."""
    pools = asset_pools or DEFAULT_ASSET_POOLS
    lid_train = set(pools["lid_variants"]["train"])
    lid_test = set(pools["lid_variants"]["test"])
    st = pools["scene_seed_pools"]
    seed_disjoint = st["train"]["seed_hi"] < st["test"]["seed_lo"] or st["test"]["seed_hi"] < st["train"]["seed_lo"]
    return {
        "lid_variant_train_test_disjoint": lid_train.isdisjoint(lid_test),
        "scene_seed_train_test_disjoint": bool(seed_disjoint),
    }
