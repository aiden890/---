"""Portable serialization for the canonical GRPO trainer's buffered rollout store."""
from __future__ import annotations

import copy
import hashlib
import io
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from async_policy import (
    validate_policy_identity,
    validate_policy_lag,
    validate_trajectory_policy,
)

STORE_SCHEMA = "grpo-trainer-store-v2"


def _allowed_path(path: Path, allowed_root: Path | None) -> Path:
    path = Path(path)
    root = Path(allowed_root).resolve() if allowed_root is not None else path.parent.resolve()
    resolved_parent = path.parent.resolve()
    if not resolved_parent.is_relative_to(root):
        raise ValueError(f"rollout-store path is outside allowed root: {path}")
    if path.suffix != ".pt":
        raise ValueError("rollout-store path must end in .pt")
    if path.is_symlink():
        raise ValueError(f"rollout-store path must not be a symlink: {path}")
    return path


def export_rollout_store(store: dict, trajectory_ids: list[str], path: Path,
                         allowed_root: Path | None = None, *, actor_id: str = "local",
                         group_id: str = "default", policy_version: int = 0,
                         policy_hash: str = "initial") -> dict[str, Any]:
    ids = [str(item) for item in trajectory_ids]
    missing = [item for item in ids if item not in store]
    if missing:
        raise KeyError(f"missing rollout trajectories: {missing}")
    policy_version, policy_hash = validate_policy_identity(policy_version, policy_hash)
    trajectories = {item: copy.deepcopy(store[item]) for item in ids}
    validate_trajectory_policy(trajectories, policy_version=policy_version,
                               policy_hash=policy_hash)
    for chunks in trajectories.values():
        for chunk in chunks:
            chunk.setdefault("policy_version", policy_version)
            chunk.setdefault("policy_hash", policy_hash)
    payload = {
        "schema": STORE_SCHEMA,
        "optimizer_update_requested": False,
        "actor_id": str(actor_id),
        "group_id": str(group_id),
        "policy_version": policy_version,
        "policy_hash": policy_hash,
        "trajectories": trajectories,
    }
    path = _allowed_path(path, allowed_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite rollout-store payload: {path}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    with os.fdopen(fd, "wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema": STORE_SCHEMA,
        "path": str(path),
        "sha256": sha256,
        "trajectory_ids": ids,
        "n_chunks": sum(len(payload["trajectories"][item]) for item in ids),
        "actor_id": payload["actor_id"],
        "group_id": payload["group_id"],
        "policy_version": policy_version,
        "policy_hash": policy_hash,
        "optimizer_update_requested": False,
    }


def import_rollout_store(store: dict, path: Path, *, expected_sha256: str,
                         allowed_root: Path | None = None,
                         learner_policy_version: int | None = None,
                         max_policy_lag: int = 1) -> dict[str, Any]:
    path = _allowed_path(path, allowed_root)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        blob = stream.read()
    actual_sha256 = hashlib.sha256(blob).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"rollout-store hash mismatch: {path}")
    payload = torch.load(io.BytesIO(blob), weights_only=True)
    if payload.get("schema") != STORE_SCHEMA:
        raise ValueError(f"unsupported rollout store schema: {payload.get('schema')!r}")
    if payload.get("optimizer_update_requested") is not False:
        raise ValueError("collector payload must not request an optimizer update")
    policy_version, policy_hash = validate_policy_identity(
        payload.get("policy_version"), payload.get("policy_hash"))
    policy_lag = None
    if learner_policy_version is not None:
        policy_lag = validate_policy_lag(
            rollout_version=policy_version, learner_version=learner_policy_version,
            max_policy_lag=max_policy_lag)
    trajectories = payload.get("trajectories", {})
    if not isinstance(trajectories, dict) or not trajectories:
        raise ValueError("rollout-store trajectories must be a non-empty mapping")
    validate_trajectory_policy(trajectories, policy_version=policy_version,
                               policy_hash=policy_hash)
    duplicates = sorted(set(map(str, trajectories)) & set(map(str, store)))
    if duplicates:
        raise RuntimeError(f"rollout trajectory already exists in trainer store: {duplicates}")
    staged = {str(trajectory_id): chunks for trajectory_id, chunks in trajectories.items()}
    for trajectory_id, chunks in staged.items():
        store[str(trajectory_id)] = chunks
    return {
        "schema": STORE_SCHEMA,
        "imported_trajectory_ids": list(map(str, trajectories)),
        "n_chunks": sum(len(chunks) for chunks in trajectories.values()),
        "actor_id": str(payload.get("actor_id", "unknown")),
        "group_id": str(payload.get("group_id", "default")),
        "policy_version": policy_version,
        "policy_hash": policy_hash,
        "policy_lag": policy_lag,
        "optimizer_update_requested": False,
    }
