#!/usr/bin/env python3
"""Consume collector payloads through the existing GRPO trainer update/save/load path."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import pickle
import socket
import struct
import time
from pathlib import Path
from typing import Optional


def _batch_group_id(item: dict) -> str:
    """Return an epoch-qualified group identity for accumulated rollout waves."""
    group_id = str(item.get("group_id", "default"))
    epoch = item.get("rollout_epoch")
    return group_id if epoch is None else f"rollout-{int(epoch)}/{group_id}"


def load_collector_payloads(run_dir: Path) -> list[dict]:
    run_dir = Path(run_dir).resolve()
    rows = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text().splitlines()
            if line.strip()]
    payloads = []
    seen_trajectory_ids: set[str] = set()
    for row in rows:
        payload = dict(row["trainer_payload"])
        if payload.get("optimizer_update_requested") is not False:
            raise ValueError("collector payload requested an optimizer update")
        payload_path = Path(payload["path"]).resolve()
        try:
            payload_path.relative_to(run_dir)
        except ValueError:
            raise ValueError(f"collector payload is outside run directory: {payload_path}")
        if not payload_path.is_file():
            raise ValueError(f"collector payload is missing: {payload_path}")
        actual_hash = hashlib.sha256(payload_path.read_bytes()).hexdigest()
        if actual_hash != payload.get("sha256"):
            raise ValueError(f"collector payload hash mismatch: {payload_path}")
        reward = float(row["reward"])
        if not math.isfinite(reward):
            raise ValueError(f"collector reward is not finite: {reward}")
        success = row.get("success")
        if success is not None and not isinstance(success, bool):
            raise ValueError(f"collector success must be boolean or null: {success!r}")
        trajectory_ids = list(map(str, payload["trajectory_ids"]))
        if not trajectory_ids:
            raise ValueError("collector payload has no trajectory IDs")
        duplicates = seen_trajectory_ids.intersection(trajectory_ids)
        if duplicates or len(trajectory_ids) != len(set(trajectory_ids)):
            raise ValueError(f"duplicate collector trajectory IDs: {sorted(duplicates)}")
        seen_trajectory_ids.update(trajectory_ids)
        payloads.append({"path": str(payload_path), "sha256": actual_hash,
                         "reward": reward,
                         "success": success,
                         "steps": int(row.get("steps", 0)),
                         "trajectory_ids": trajectory_ids,
                         "actor_id": str(payload.get("actor_id", "unknown")),
                         "group_id": str(payload.get("group_id", "default")),
                         "rollout_epoch": (
                             None if row.get("rollout_epoch",
                                             payload.get("rollout_epoch")) is None
                             else int(row.get("rollout_epoch",
                                              payload.get("rollout_epoch")))),
                         "policy_version": int(payload.get("policy_version", 0)),
                         "policy_hash": str(payload.get("policy_hash", "initial"))})
    if len(payloads) < 2:
        raise ValueError("consume smoke requires at least two collected trajectories")
    return payloads


def validate_rollout_batch(payloads: list[dict], *, expected_group_size: Optional[int] = None,
                           min_rollout_epochs: Optional[int] = None) -> dict:
    """Validate collection-level Z-1 batching separately from optimizer epochs.

    ``rollout_epoch`` is collection provenance: each value denotes a fresh rollout
    wave. It is deliberately unrelated to the number of times the optimizer reuses
    the resulting batch.
    """
    groups: dict[str, list[dict]] = {}
    seen_trajectory_ids: set[str] = set()
    for item in payloads:
        groups.setdefault(_batch_group_id(item), []).append(item)
        ids = list(map(str, item.get("trajectory_ids", [])))
        duplicates = seen_trajectory_ids.intersection(ids)
        if duplicates or len(ids) != len(set(ids)):
            raise ValueError(f"duplicate rollout trajectory IDs: {sorted(duplicates)}")
        seen_trajectory_ids.update(ids)
    sizes = {group_id: sum(len(item.get("trajectory_ids", [])) for item in items)
             for group_id, items in groups.items()}
    if expected_group_size is not None:
        expected_group_size = int(expected_group_size)
        if expected_group_size <= 0:
            raise ValueError("expected_group_size must be positive")
        bad = {group_id: size for group_id, size in sizes.items()
               if size != expected_group_size}
        if bad:
            raise ValueError(
                f"GRPO groups must contain exactly {expected_group_size} trajectories: {bad}")

    labelled = [item.get("rollout_epoch") for item in payloads]
    rollout_epochs = sorted({int(value) for value in labelled if value is not None})
    if min_rollout_epochs is not None:
        min_rollout_epochs = int(min_rollout_epochs)
        if min_rollout_epochs <= 0:
            raise ValueError("min_rollout_epochs must be positive")
        if any(value is None for value in labelled):
            raise ValueError(
                "rollout_epoch provenance is required when min_rollout_epochs is configured")
        if len(rollout_epochs) < min_rollout_epochs:
            raise ValueError(
                f"need at least {min_rollout_epochs} fresh rollout epochs before update; "
                f"found {len(rollout_epochs)}")
    return {
        "group_count_collected": len(groups),
        "group_sizes": sizes,
        "rollout_epochs_collected": len(rollout_epochs),
        "rollout_epoch_ids": rollout_epochs,
    }


def inspect_action_chunks(payloads: list[dict], advantages: dict[str, float]) -> dict:
    """Count actual serialized chunks, including the subset carrying gradients.

    Payload metadata is not trusted for this gate. Files have already passed SHA-256
    validation in :func:`load_collector_payloads`; load them with ``weights_only`` and
    inspect the canonical trajectory mapping. The trainer ignores trajectories with
    zero advantage, so only chunks from non-zero-advantage trajectories are trainable.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - production image always provides torch
        raise RuntimeError("PyTorch is required to inspect rollout payload chunks") from exc

    total = 0
    trainable = 0
    by_group: dict[str, dict[str, int]] = {}
    by_trajectory: dict[str, dict[str, object]] = {}
    for item in payloads:
        path = Path(item["path"])
        blob = path.read_bytes()
        actual_hash = hashlib.sha256(blob).hexdigest()
        if actual_hash != item["sha256"]:
            raise ValueError(f"collector payload hash changed before chunk inspection: {path}")
        payload = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
        if payload.get("schema") != "grpo-trainer-store-v2":
            raise ValueError(f"unsupported rollout store schema: {payload.get('schema')!r}")
        trajectories = payload.get("trajectories")
        if not isinstance(trajectories, dict) or not trajectories:
            raise ValueError(f"rollout payload has no trajectory mapping: {path}")
        expected_ids = set(map(str, item["trajectory_ids"]))
        actual_ids = set(map(str, trajectories))
        if actual_ids != expected_ids:
            raise ValueError(
                f"rollout payload trajectory IDs do not match metadata: {path}")
        group = by_group.setdefault(_batch_group_id(item), {"total": 0, "trainable": 0})
        for trajectory_id, chunks in trajectories.items():
            trajectory_id = str(trajectory_id)
            if not isinstance(chunks, (list, tuple)):
                raise ValueError(f"trajectory chunks must be a sequence: {trajectory_id}")
            count = len(chunks)
            advantage = float(advantages.get(trajectory_id, 0.0))
            is_trainable = advantage != 0.0
            total += count
            group["total"] += count
            if is_trainable:
                trainable += count
                group["trainable"] += count
            by_trajectory[trajectory_id] = {
                "chunks": count, "advantage": advantage, "trainable": is_trainable}
    return {
        "action_chunks_total": total,
        "trainable_action_chunks": trainable,
        "action_chunks_by_group": by_group,
        "action_chunks_by_trajectory": by_trajectory,
    }


def optimizer_update_epochs(cfg: dict) -> int:
    """Resolve optimizer reuse count without conflating it with rollout epochs."""
    explicit = cfg.get("optimizer_update_epochs")
    legacy = cfg.get("update_epochs")
    if explicit is not None and legacy is not None and int(explicit) != int(legacy):
        raise ValueError("optimizer_update_epochs conflicts with legacy update_epochs")
    value = int(explicit if explicit is not None else (legacy if legacy is not None else 2))
    if value <= 0:
        raise ValueError("optimizer_update_epochs must be positive")
    return value


def require_trainable_action_chunks(diagnostics: dict, minimum: int) -> None:
    minimum = int(minimum)
    if minimum < 0:
        raise ValueError("min_trainable_action_chunks must be non-negative")
    actual = int(diagnostics.get("trainable_action_chunks", 0))
    if actual < minimum:
        raise ValueError(
            f"need at least {minimum} trainable action chunks before update; found {actual}")


def build_smoke_advantages(payloads: list[dict], *, success_decay_gamma: Optional[float] = None,
                           action_chunk_steps: int = 16) -> dict[str, float]:
    """Build group-relative advantages, optionally using Z-1 success calibration.

    The stored environment reward remains the exact sparse 0/1 grasp predicate.  When
    ``success_decay_gamma`` is set, successful returns are calibrated as gamma**d where
    d is the number of executed action chunks. Mixed groups are standardized normally;
    all-success groups use Z-1's non-negative minimum baseline.
    """
    if success_decay_gamma is not None and not 0.0 < success_decay_gamma <= 1.0:
        raise ValueError("success_decay_gamma must be in (0, 1]")
    if action_chunk_steps <= 0:
        raise ValueError("action_chunk_steps must be positive")
    advantages = {}
    groups: dict[str, list[dict]] = {}
    for item in payloads:
        groups.setdefault(_batch_group_id(item), []).append(item)
    for items in groups.values():
        rewards = [float(item["reward"]) for item in items]
        successes = [item.get("success") for item in items]
        if len(set(rewards)) < 2 and all(value is not None for value in successes):
            rewards = [float(bool(value)) for value in successes]
        if success_decay_gamma is not None and all(value is not None for value in successes):
            rewards = [
                (float(success_decay_gamma) ** int(math.ceil(item.get("steps", 0) /
                                                              action_chunk_steps))
                 if bool(success) else 0.0)
                for item, success in zip(items, successes)
            ]
        all_success = bool(successes) and all(value is True for value in successes)
        if success_decay_gamma is not None and all_success:
            baseline = min(rewards)
            values = [value - baseline for value in rewards]
            for item, value in zip(items, values):
                for trajectory_id in item["trajectory_ids"]:
                    advantages[str(trajectory_id)] = float(value)
            continue
        mean = sum(rewards) / len(rewards)
        variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
        std = math.sqrt(variance)
        if std > 1e-8:
            values = [(value - mean) / std for value in rewards]
        else:
            # Bounded mutation smoke fallback. Production collection can discard these
            # zero-variance groups before import via discard_homogeneous_groups.
            values = [-1.0 if index % 2 == 0 else 1.0 for index in range(len(rewards))]
        for item, value in zip(items, values):
            for trajectory_id in item["trajectory_ids"]:
                advantages[str(trajectory_id)] = float(value)
    return advantages


def filter_homogeneous_groups(payloads: list[dict], *, retain_all_success: bool = False
                              ) -> tuple[list[dict], list[str]]:
    """Drop outcome-homogeneous groups, falling back to reward variance.

    Simulator success is the authoritative GRPO outcome when it is present for
    every trajectory.  Shaped rewards must not accidentally retain an all-win
    or all-loss group.  Older payloads without complete success metadata use
    reward variance for backward compatibility.
    """
    groups: dict[str, list[dict]] = {}
    for item in payloads:
        groups.setdefault(_batch_group_id(item), []).append(item)
    kept, dropped = [], []
    for group_id, items in groups.items():
        rewards = {float(item["reward"]) for item in items}
        success_values = [item.get("success") for item in items]
        if all(value is not None for value in success_values):
            all_success = bool(success_values) and all(bool(value) for value in success_values)
            homogeneous = len({bool(value) for value in success_values}) < 2
            if all_success and retain_all_success:
                homogeneous = False
        else:
            homogeneous = len(rewards) < 2
        if homogeneous:
            dropped.append(group_id)
        else:
            kept.extend(items)
    return kept, dropped


def validate_update(update: dict) -> dict:
    epoch_stats = list(update.get("epoch_stats") or [])
    epoch0_ratio = epoch_stats[0].get("mean_ratio") if epoch_stats else None
    policy_lag = int(update.get("policy_lag", 0))
    nonfinite = int(update.get("n_nonfinite", 0)) + int(update.get("post_step_n_nonfinite", 0))
    dropped = int(update.get("n_dropped", 0)) + int(update.get("post_step_n_dropped", 0))
    checks = {
        "new_update_not_replay": not bool(update.get("replayed_update", False)),
        "adapter_delta_positive": float(update.get("adapter_delta_l2", 0.0)) > 0.0,
        "no_nonfinite": nonfinite == 0,
        "no_dropped": dropped == 0,
        # Lag-0 must reproduce the behaviour policy exactly. Lag-1 is intentionally
        # off-policy, so require a finite positive ratio instead of equality to one.
        "epoch0_ratio_valid": (
            epoch0_ratio is not None and
            # The behavior policy may run on GB10 while the learner recomputes
            # log-probabilities on RTX 3090. Allow normal cross-architecture BF16 drift,
            # while still rejecting a materially different lag-0 policy.
            (abs(float(epoch0_ratio) - 1.0) <= 1e-3 if policy_lag == 0
             else math.isfinite(float(epoch0_ratio)) and float(epoch0_ratio) > 0.0)
        ),
    }
    return {"pass": all(checks.values()), "checks": checks,
            "epoch0_mean_ratio": epoch0_ratio, "policy_lag": policy_lag,
            "nonfinite": nonfinite, "dropped": dropped}


class TrainerRPC:
    def __init__(self, host: str, port: int):
        self.sock = socket.create_connection((host, port), timeout=60)
        # The connect timeout must not become a lifetime timeout for a real GPU
        # update. A full group can contain hundreds of flow chunks and legitimately
        # take longer than 60 seconds on the learner. Once connected, wait for the
        # trainer's length-prefixed response; update IDs provide retry safety if the
        # connection itself is lost.
        self.sock.settimeout(None)

    @staticmethod
    def _recv_exact(sock, size):
        data = bytearray()
        while len(data) < size:
            packet = sock.recv(size - len(data))
            if not packet:
                raise EOFError(f"trainer closed after {len(data)}/{size} bytes")
            data.extend(packet)
        return bytes(data)

    def call(self, request):
        blob = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        self.sock.sendall(struct.pack(">I", len(blob)) + blob)
        size = struct.unpack(">I", self._recv_exact(self.sock, 4))[0]
        response = pickle.loads(self._recv_exact(self.sock, size))
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(response["error"])
        return response

    def close(self):
        self.sock.close()


def consume_smoke(run_dir: Path, host: str, port: int, cfg: dict) -> dict:
    run_dir = Path(run_dir)
    audit_path = run_dir / "audit.json"
    report_path = run_dir / "consume_smoke" / "consume_smoke.json"
    if not (run_dir / "DONE").is_file() or not audit_path.is_file():
        raise RuntimeError("collection DONE and audit.json are required before consume smoke")
    if not json.loads(audit_path.read_text(encoding="utf-8")).get("pass", False):
        raise RuntimeError("collector audit did not pass")
    if report_path.exists():
        raise RuntimeError(f"consume smoke already has a terminal report: {report_path}")
    payloads = load_collector_payloads(run_dir)
    collection_diagnostics = validate_rollout_batch(
        payloads,
        expected_group_size=cfg.get("expected_group_size"),
        min_rollout_epochs=cfg.get("min_rollout_epochs"),
    )
    dropped_groups = []
    success_decay_gamma = cfg.get("success_decay_gamma")
    action_chunk_steps = int(cfg.get("action_chunk_steps", 16))
    if bool(cfg.get("discard_homogeneous_groups", False)):
        payloads, dropped_groups = filter_homogeneous_groups(
            payloads, retain_all_success=success_decay_gamma is not None)
        if not payloads:
            raise ValueError("all collected groups were homogeneous; no gradient update needed")
    policy_snapshots = {(item["policy_version"], item["policy_hash"]) for item in payloads}
    if len(policy_snapshots) != 1:
        raise ValueError(
            f"consume smoke requires one policy snapshot: {sorted(policy_snapshots)}")
    advantages = build_smoke_advantages(
        payloads, success_decay_gamma=success_decay_gamma,
        action_chunk_steps=action_chunk_steps)
    chunk_diagnostics = inspect_action_chunks(payloads, advantages)
    minimum_chunks = int(cfg.get("min_trainable_action_chunks", 0))
    require_trainable_action_chunks(chunk_diagnostics, minimum_chunks)
    update_epochs = optimizer_update_epochs(cfg)
    checkpoint = run_dir / "consume_smoke" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    client = TrainerRPC(host, port)
    started = time.time()
    try:
        phase_started = time.time()
        reset = client.call({"op": "reset"})
        reset_seconds = time.time() - phase_started
        phase_started = time.time()
        imports = [client.call({"op": "import_store", "path": item["path"],
                                "expected_sha256": item["sha256"],
                                "max_policy_lag": 1})
                   for item in payloads]
        import_seconds = time.time() - phase_started
        before = client.call({"op": "metrics"})
        phase_started = time.time()
        target_kl = cfg.get("target_kl", 0.02)
        update = client.call({
            "op": "update", "advantages": advantages,
            "clip": float(cfg.get("clip", 0.1)),
            "kl_coef": float(cfg.get("kl_coef", 0.005)),
            "ratio_max": float(cfg.get("ratio_max", 10.0)),
            "adv_clip": float(cfg.get("adv_clip", 3.0)),
            "update_epochs": update_epochs,
            "target_kl": (None if target_kl is None else float(target_kl)),
            "update_id": f"collector-consume-{run_dir.name}",
            "curriculum_state": None,
        })
        update_seconds = time.time() - phase_started
        phase_started = time.time()
        update_index = int(update.get("policy_version_after", 1))
        saved = client.call({"op": "save", "path": str(checkpoint),
                             "update_index": update_index,
                             "train_meta": {"source": "rlinf_grid_consume_smoke"}})
        loaded = client.call({"op": "load", "path": str(checkpoint)})
        checkpoint_seconds = time.time() - phase_started
    finally:
        client.close()
    gate = validate_update(update)
    checkpoint_ok = checkpoint.is_file() and bool(saved) and bool(loaded)
    report = {
        "status": "PASS" if gate["pass"] and checkpoint_ok else "FAIL",
        "run_dir": str(run_dir),
        "payload_count": len(payloads),
        "group_count": len({_batch_group_id(item) for item in payloads}),
        "dropped_homogeneous_groups": dropped_groups,
        "trajectory_count": len(advantages),
        "advantages": advantages,
        "algorithm_diagnostics": {
            "collection": collection_diagnostics,
            "chunks": chunk_diagnostics,
            "minimum_trainable_action_chunks": minimum_chunks,
            "success_decay_gamma": success_decay_gamma,
            "action_chunk_steps": action_chunk_steps,
            "groups": {
                group_id: [{
                    "trajectory_ids": item["trajectory_ids"],
                    "success": item.get("success"),
                    "sparse_reward": item["reward"],
                    "steps": item.get("steps"),
                    "completion_chunks": int(math.ceil(item.get("steps", 0) /
                                                       action_chunk_steps)),
                    "advantages": [advantages[tid] for tid in item["trajectory_ids"]],
                } for item in payloads if _batch_group_id(item) == group_id]
                for group_id in sorted({_batch_group_id(item) for item in payloads})
            },
            "objective": {
                "clip": float(cfg.get("clip", 0.1)),
                "kl_coef": float(cfg.get("kl_coef", 0.005)),
                "optimizer_update_epochs": update_epochs,
                "rollout_epochs": collection_diagnostics["rollout_epochs_collected"],
                "target_kl": target_kl,
            },
        },
        "reset": reset,
        "imports": imports,
        "store_before_update": before,
        "update": update,
        "update_gate": gate,
        "saved": saved,
        "loaded": loaded,
        "checkpoint_reloadable": checkpoint_ok,
        "checkpoint": str(checkpoint),
        "timing": {"reset_seconds": reset_seconds,
                   "import_seconds": import_seconds,
                   "gradient_update_seconds": update_seconds,
                   "checkpoint_roundtrip_seconds": checkpoint_seconds},
        "elapsed_seconds": time.time() - started,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10088)
    parser.add_argument("--config", default="/integration/configs/grid_smoke.yaml")
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(Path(args.config).read_text()).get("consume_smoke", {})
    report = consume_smoke(Path(args.run_dir), args.host, args.port, config)
    print(json.dumps(report, indent=2, default=str))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
