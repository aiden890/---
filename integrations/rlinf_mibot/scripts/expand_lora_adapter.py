#!/usr/bin/env python3
"""Expand a published qkv LoRA actor snapshot into an all-linear snapshot.

The target template must be published by a freshly initialized all-linear trainer.
Existing tensors are copied from the source snapshot; newly introduced LoRA modules
retain the template's random A and zero B initialization.  Optimizer state is
intentionally not migrated because published actor snapshots do not contain it and the
parameter layout changed.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

import torch


SCHEMA = "grpo-adapter-v1"


def mutable_policy_hash(lora: dict[str, torch.Tensor],
                        extra: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted({**lora, **extra}.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def expand_snapshot(source: dict, template: dict) -> dict:
    if source.get("schema") != SCHEMA or template.get("schema") != SCHEMA:
        raise ValueError("source and template must be grpo-adapter-v1 snapshots")
    source_lora = source.get("lora") or {}
    target_lora = template.get("lora") or {}
    if not source_lora or not target_lora:
        raise ValueError("source and template must contain LoRA tensors")
    missing = sorted(set(source_lora) - set(target_lora))
    if missing:
        raise ValueError(f"source tensors absent from target layout: {missing[:3]}")

    merged = {name: tensor.detach().cpu().clone()
              for name, tensor in target_lora.items()}
    for name, tensor in source_lora.items():
        if tuple(tensor.shape) != tuple(merged[name].shape):
            raise ValueError(
                f"shape mismatch for {name}: source={tuple(tensor.shape)} "
                f"target={tuple(merged[name].shape)}")
        merged[name] = tensor.detach().cpu().clone()

    new_b = [name for name in merged
             if name not in source_lora and ".lora_B." in name]
    nonzero_b = [name for name in new_b if torch.count_nonzero(merged[name]).item()]
    if nonzero_b:
        raise ValueError(f"new LoRA-B tensors must be zero initialized: {nonzero_b[:3]}")

    source_extra = source.get("extra_trainable") or {}
    target_extra = template.get("extra_trainable") or {}
    if source_extra:
        raise ValueError("qkv source unexpectedly contains extra trainable tensors")
    extra = {name: tensor.detach().cpu().clone()
             for name, tensor in target_extra.items()}
    policy_hash = mutable_policy_hash(merged, extra)
    return {
        "schema": SCHEMA,
        "lora": merged,
        "extra_trainable": extra,
        "policy_version": int(source["policy_version"]),
        "policy_hash": policy_hash,
        "migration": {
            "kind": "qkv-to-all-linear",
            "source_policy_hash": str(source["policy_hash"]),
            "copied_tensors": len(source_lora),
            "initialized_tensors": len(merged) - len(source_lora),
            "optimizer_state_reset": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = torch.load(args.source, map_location="cpu", weights_only=True)
    template = torch.load(args.template, map_location="cpu", weights_only=True)
    result = expand_snapshot(source, template)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent)
    os.close(fd)
    try:
        torch.save(result, temporary)
        os.replace(temporary, args.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print({key: result[key] for key in ("policy_version", "policy_hash", "migration")})


if __name__ == "__main__":
    main()
