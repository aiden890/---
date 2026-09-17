"""Strict roundtrip gate for a real all-linear GRPO checkpoint.

Loads the requested checkpoint into the real Xiaomi model, records every named LoRA
tensor and a deterministic action, corrupts every LoRA-B tensor, reloads the same
checkpoint, and requires bit-exact parameter and action restoration. The corruption
must change both the parameters and the action, so neither check can pass trivially.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, "/rl_env/src")
sys.path.insert(0, "/train/src")
sys.path.insert(0, "/train/scripts")
from audit_p0_verify import build_inputs  # noqa: E402
from grpo_trainer_server import GRPOTrainerServer  # noqa: E402
from lora import lora_state_dict  # noqa: E402


def server_args(model: str) -> SimpleNamespace:
    """Match corrected_exact_3154248_micro_lr1e6 checkpoint metadata."""
    return SimpleNamespace(
        model=model, host="127.0.0.1", port=0,
        lr=1e-6, optimizer="adamw", weight_decay=0.01,
        train_mode="adapter_only", expert_lr=None,
        rank=8, alpha=32, adapter_skills="grasp", lora_targets="all_linear",
        num_steps=5, sampler="pirl", eta=0.1, replan_steps=16,
        real_action_dim=12, clip=0.2, kl_coef=0.0, ratio_max=10.0,
        adv_clip=3.0, grad_clip=1.0, update_epochs=2, target_kl=0.05,
        grad_checkpoint=True, anchor_coef=0.0,
        train_source_manifest="/train/source_manifest.json",
        env_source_manifest="/rl_env/source_manifest.json",
        source_commit="roundtrip-verifier", source_manifest_sha256="roundtrip-verifier",
    )


def deterministic_action(server, inputs, seed: int) -> np.ndarray:
    response = server.op_sample({
        "inputs": {key: (value.clone() if isinstance(value, torch.Tensor) else value)
                   for key, value in inputs.items()},
        "eta": 0.0,
        "skill": "grasp",
        "seed": seed,
        "chunk_index": 0,
    })
    return response["actions"].float().numpy()


def snapshot_lora(server) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in lora_state_dict(server.wrappers).items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--robot-type", default="robocasa365")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if checkpoint.parent.name == "out":
        raise RuntimeError(f"shared results/out checkpoint is forbidden: {checkpoint}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    from transformers import AutoProcessor

    server = GRPOTrainerServer(server_args(args.model))
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    inputs = build_inputs(processor, args.robot_type)

    first_load = server.op_load({"path": str(checkpoint)})
    reference_params = snapshot_lora(server)
    reference_action = deterministic_action(server, inputs, args.seed)

    with torch.no_grad():
        for wrapper in server.wrappers:
            wrapper.lora_B["grasp"].add_(torch.full_like(wrapper.lora_B["grasp"], 0.01))
    corrupted_params = snapshot_lora(server)
    corrupted_action = deterministic_action(server, inputs, args.seed)

    changed_param_count = sum(
        not torch.equal(reference_params[name], corrupted_params[name])
        for name in reference_params
    )
    action_corruption_max_abs = float(np.max(np.abs(reference_action - corrupted_action)))

    second_load = server.op_load({"path": str(checkpoint)})
    restored_params = snapshot_lora(server)
    restored_action = deterministic_action(server, inputs, args.seed)

    mismatched_params = [
        name for name in reference_params
        if not torch.equal(reference_params[name], restored_params[name])
    ]
    action_restore_max_abs = float(np.max(np.abs(reference_action - restored_action)))

    checks = {
        "run_unique_checkpoint_path": checkpoint.parent.name != "out",
        "strict_named_layout": first_load.get("n_tensors") == len(reference_params) == 378,
        "mutation_changed_parameters": changed_param_count > 0,
        "mutation_changed_action": action_corruption_max_abs > 0.0,
        "parameters_restored_bit_exact": not mismatched_params,
        "action_restored_bit_exact": action_restore_max_abs == 0.0,
        "second_strict_load_matches_first": second_load.get("n_tensors") == first_load.get("n_tensors"),
    }
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "first_load": first_load,
        "second_load": second_load,
        "n_named_lora_tensors": len(reference_params),
        "changed_param_count": changed_param_count,
        "mismatched_params_after_reload": mismatched_params[:10],
        "action_corruption_max_abs": action_corruption_max_abs,
        "action_restore_max_abs": action_restore_max_abs,
        "checks": checks,
        "overall_pass": all(checks.values()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
