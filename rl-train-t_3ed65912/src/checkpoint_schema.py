"""Strict, stable metadata schema for named LoRA checkpoints."""
from __future__ import annotations

SCHEMA_VERSION = 2


def build_checkpoint_metadata(*, adapter_skills, targets, rank, alpha, base_model,
                              sampler, config, source_commit,
                              source_manifest_sha256, update_index):
    return {
        "schema_version": SCHEMA_VERSION,
        "adapter_skills": list(adapter_skills),
        "targets": list(targets),
        "rank": int(rank),
        "alpha": int(alpha),
        "base_model": str(base_model),
        "sampler": str(sampler),
        "config": dict(config),
        "source_commit": str(source_commit),
        "source_manifest_sha256": str(source_manifest_sha256),
        "update_index": int(update_index),
    }


def validate_checkpoint_metadata(metadata, *, adapter_skills, targets, rank, alpha):
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported checkpoint schema; migrate legacy checkpoints explicitly")
    expected = {
        "adapter_skills": list(adapter_skills),
        "targets": list(targets),
        "rank": int(rank),
        "alpha": int(alpha),
    }
    mismatches = {key: {"expected": value, "actual": metadata.get(key)}
                  for key, value in expected.items() if metadata.get(key) != value}
    required = ("base_model", "sampler", "config", "source_commit",
                "source_manifest_sha256", "update_index")
    missing = [key for key in required if key not in metadata]
    if mismatches or missing:
        raise RuntimeError(f"checkpoint metadata mismatch: mismatches={mismatches} missing={missing}")
    return metadata
