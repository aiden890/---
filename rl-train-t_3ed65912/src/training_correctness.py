"""Pure correctness helpers shared by the GRPO client and trainer."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence


REWARD_COMPONENTS = (
    "skill_terminal", "official_terminal", "hold_stay", "hold_drift",
    "hold_lapse", "drop", "collision", "timeout",
    "milestone", "approach",
)


@dataclass
class RewardComponents:
    skill_terminal: float = 0.0
    official_terminal: float = 0.0
    hold_stay: float = 0.0
    hold_drift: float = 0.0
    hold_lapse: float = 0.0
    drop: float = 0.0
    collision: float = 0.0
    timeout: float = 0.0
    milestone: float = 0.0
    approach: float = 0.0

    def total(self) -> float:
        values = asdict(self)
        return sum(values[k] for k in REWARD_COMPONENTS)

    def as_dict(self) -> dict[str, float]:
        out = asdict(self)
        out["total"] = self.total()
        return out


def skill_completion_reward(
    completion_step: Optional[int], success_reward: float = 1.0,
    gamma: float = 0.998, decay_enabled: bool = True,
) -> float:
    if completion_step is None:
        return 0.0
    if completion_step < 0:
        raise ValueError("completion_step must be non-negative")
    if not (0.0 < gamma <= 1.0):
        raise ValueError("gamma must be in (0, 1]")
    return float(success_reward) * (float(gamma) ** int(completion_step) if decay_enabled else 1.0)


def nonduplicated_skill_reward(official_terminal: float, completion_step: Optional[int],
                               success_reward: float, gamma: float,
                               decay_enabled: bool) -> float:
    """Avoid paying the same PLACE terminal through both official and skill channels."""
    if float(official_terminal) != 0.0:
        return 0.0
    return skill_completion_reward(
        completion_step, success_reward, gamma, decay_enabled)


def skill_timeout_reward(outcome_name: str, timeout_penalty: float) -> float:
    return -float(timeout_penalty) if str(outcome_name) == "TIMEOUT" else 0.0


def hold_enabled_for_variant(reward_variant: str, hold_steps: int) -> bool:
    """The terminal-only ablation must not silently include hold shaping."""
    return str(reward_variant) != "simulator_terminal_only" and int(hold_steps) > 0


def skill_terminal_enabled_for_variant(reward_variant: str) -> bool:
    """Task-success binary reward must not pay intermediate skill completion."""
    return str(reward_variant) != "simulator_terminal_only"


def reset_gated_store(client, defer_update: bool) -> bool:
    """Clear a skipped single-group rollout; preserve prior groups in batched collection."""
    if defer_update:
        return False
    client.reset_store()
    return True


class ExactHoldWindow:
    """Latch first success and stop after exactly N subsequent environment steps."""

    def __init__(self, hold_steps: int):
        if int(hold_steps) < 0:
            raise ValueError("hold_steps must be non-negative")
        self.hold_steps = int(hold_steps)
        self.latched = False
        self.held_steps = 0
        self.lapse_steps = 0

    def observe(self, success_now: bool, terminated: bool = False) -> bool:
        if not self.latched:
            if not success_now:
                return bool(terminated)
            self.latched = True
            return self.hold_steps == 0 or bool(terminated)
        if not success_now:
            self.lapse_steps += 1
        self.held_steps += 1
        return bool(terminated) or self.held_steps >= self.hold_steps


def validate_optimizer_config(update_epochs: int, ratio_max: float, clip: float,
                              grad_clip: float, eta: float):
    values = (int(update_epochs), float(ratio_max), float(clip),
              float(grad_clip), float(eta))
    epochs, ratio, clip_value, grad, noise = values
    if epochs < 1:
        raise ValueError("update_epochs must be >= 1")
    if ratio <= 1.0:
        raise ValueError("ratio_max must be > 1")
    if clip_value <= 0.0:
        raise ValueError("clip must be positive")
    if grad <= 0.0:
        raise ValueError("grad_clip must be positive")
    if noise < 0.0:
        raise ValueError("eta must be non-negative")
    return values


def mean_loss_scale(n_chunks: int) -> float:
    if int(n_chunks) <= 0:
        raise ValueError("n_chunks must be positive")
    return 1.0 / int(n_chunks)


def guarded_ratio(raw_logratio: float, ratio_max: float) -> Optional[float]:
    """Reject raw excursions before the numerical exp clamp can hide them."""
    raw = float(raw_logratio)
    limit = float(ratio_max)
    if not math.isfinite(raw) or not math.isfinite(limit) or limit <= 1.0:
        return None
    bound = math.log(limit)
    if raw < -bound or raw > bound:
        return None
    return math.exp(raw)


def should_stop_for_kl(epoch: int, mean_kl: float,
                       target_kl: Optional[float]) -> bool:
    return (target_kl is not None and float(target_kl) > 0.0 and int(epoch) > 0
            and float(mean_kl) > float(target_kl))


def append_progress(path: Path | str, row: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def source_manifest(root: Path | str, relative_paths: Sequence[str]) -> dict[str, str]:
    root = Path(root)
    out = {}
    for rel in relative_paths:
        p = root / rel
        out[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def verify_source_manifest(root: Path | str, expected: Mapping[str, str]) -> None:
    actual = source_manifest(root, list(expected))
    mismatched = [name for name, digest in expected.items() if actual.get(name) != digest]
    if mismatched:
        raise RuntimeError(f"source hash mismatch: {mismatched}")


def verify_deployment_manifest(root: Path | str, manifest_path: Path | str) -> dict:
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("dirty"):
        raise RuntimeError("deployment manifest is dirty")
    commit = str(manifest.get("commit") or "")
    if not commit:
        raise RuntimeError("deployment manifest has no commit")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("deployment manifest has no file hashes")
    verify_source_manifest(root, files)
    return manifest
