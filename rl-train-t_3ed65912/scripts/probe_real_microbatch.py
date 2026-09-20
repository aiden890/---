#!/usr/bin/env python3
"""Compare serial and microbatched pi-RL recompute on real rollout chunks."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, "/train/src")
from grpo_trainer_server import GRPOTrainerServer  # noqa: E402


def arguments(batch_model_forward: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        model="/checkpoint", train_source_manifest="/train/source_manifest.json",
        env_source_manifest="/rl_env/source_manifest.json", host="127.0.0.1", port=10088,
        lr=5e-6, optimizer="adamw", weight_decay=0.01, train_mode="adapter_only",
        rank=8, alpha=32, adapter_skills="grasp,move_holding,place", expert_lr=None,
        anchor_coef=0.0, lora_targets="qkv_proj", num_steps=5, sampler="pirl", eta=0.1,
        replan_steps=16, real_action_dim=12, clip=0.2, kl_coef=0.0, ratio_max=10.0,
        adv_clip=3.0, grad_clip=1.0, update_epochs=1, target_kl=None,
        grad_checkpoint=False, microbatch_size=4, post_diag_chunks=128,
        batch_model_forward=batch_model_forward,
        source_commit="probe", source_manifest_sha256="probe",
    )


def gradient_vector(server: GRPOTrainerServer) -> torch.Tensor:
    return torch.cat([
        (parameter.grad.detach().float().flatten() if parameter.grad is not None
         else torch.zeros(parameter.numel(), device=parameter.device))
        for parameter in server.trainable_params
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload_root")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--key-index", type=int, default=0)
    parser.add_argument("--collated-forward", action="store_true")
    parser.add_argument("--out", required=True)
    cli = parser.parse_args()
    server = GRPOTrainerServer(arguments(cli.collated_forward))
    loaded = server.op_load({"path": cli.checkpoint})
    candidates = []
    for path in sorted(Path(cli.payload_root).rglob("*.pt")):
        blob = torch.load(path, map_location="cpu")
        for trajectory_id, chunks in blob.get("trajectories", {}).items():
            for chunk in chunks:
                candidates.append((trajectory_id, chunk))
    groups = {}
    for item in candidates:
        groups.setdefault(server._chunk_batch_key(item[1]), []).append(item)
    eligible = sorted(
        (items for items in groups.values() if len(items) >= cli.batch),
        key=lambda items: server._chunk_batch_key(items[0][1]))
    if not eligible:
        raise RuntimeError("no compatible microbatch group found")
    compatible = eligible[cli.key_index % len(eligible)][:cli.batch]
    if len(compatible) != cli.batch:
        raise RuntimeError(f"only {len(compatible)} compatible chunks found")
    chunks = [item[1] for item in compatible]

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with torch.no_grad():
        serial_terms = [server._forward_new_logp_terms(chunk) for chunk in chunks]
    torch.cuda.synchronize()
    serial_forward_seconds = time.time() - started

    started = time.time()
    with torch.no_grad():
        batch_terms = server._forward_new_logp_terms_batch(chunks)
    torch.cuda.synchronize()
    batch_forward_seconds = time.time() - started
    term_max_abs = max(float((left - right).abs().max().cpu())
                       for left, right in zip(serial_terms, batch_terms))

    server.opt.zero_grad(set_to_none=True)
    started = time.time()
    for chunk in chunks:
        server._forward_new_logp_terms(chunk).mean().backward()
    torch.cuda.synchronize()
    serial_backward_seconds = time.time() - started
    serial_gradient = gradient_vector(server)

    server.opt.zero_grad(set_to_none=True)
    started = time.time()
    batch_grad_terms = server._forward_new_logp_terms_batch(chunks)
    torch.stack([value.mean() for value in batch_grad_terms]).sum().backward()
    torch.cuda.synchronize()
    batch_backward_seconds = time.time() - started
    batch_gradient = gradient_vector(server)
    gradient_diff = batch_gradient - serial_gradient
    serial_norm = float(torch.linalg.vector_norm(serial_gradient).cpu())
    report = {
        "status": "PASS",
        "batch": cli.batch,
        "key_index": cli.key_index,
        "compatible_key_count": len(eligible),
        "denoise_index": int(compatible[0][1]["denoise_index"]),
        "collated_forward": cli.collated_forward,
        "chunks_available": len(candidates),
        "trajectory_ids": [item[0] for item in compatible],
        "policy_version": loaded["policy_version"],
        "policy_hash": loaded["policy_hash"],
        "term_max_abs": term_max_abs,
        "gradient_l2": serial_norm,
        "gradient_diff_l2": float(torch.linalg.vector_norm(gradient_diff).cpu()),
        "gradient_relative_l2": (
            float(torch.linalg.vector_norm(gradient_diff).cpu()) / max(serial_norm, 1e-12)),
        "gradient_diff_linf": float(gradient_diff.abs().max().cpu()),
        "serial_forward_seconds": serial_forward_seconds,
        "batch_forward_seconds": batch_forward_seconds,
        "serial_forward_backward_seconds": serial_backward_seconds,
        "batch_forward_backward_seconds": batch_backward_seconds,
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
    }
    if not all(torch.isfinite(value).all() for value in batch_terms):
        report["status"] = "FAIL"
    print(json.dumps(report, indent=2))
    output = Path(cli.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
