#!/usr/bin/env python3
"""Validate the canonical multi-group GPU gate config without touching Docker/GPU."""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path


def audit_nonjoint_source(source):
    """Static guard for the latest independent piRL ratio/mask path."""
    checks = {
        "new_logprob_elementwise_masked": bool(
            re.search(r"^\s*return per\[exec_mask\]\s*$", source, re.MULTILINE)),
        "old_logprob_elementwise_masked": bool(
            re.search(r"^\s*return old\[mask\]\.to\([^\n]+\)\s*$", source, re.MULTILINE)),
        "ratio_exp_is_elementwise": (
            "logratio = new_terms - old_terms" in source and
            bool(re.search(r"^\s*ratio = torch\.exp\(valid_lr\)\s*$",
                           source, re.MULTILINE))),
    }
    if not all(checks.values()):
        raise ValueError(f"non-joint source audit failed: {checks}")
    return checks


def validate(config, run):
    trainer = config["trainer"]
    client = config["client"]
    errors = []
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run):
        errors.append("run must be a safe basename")
    if client.get("group", 0) < 2:
        errors.append("group must be >= 2")
    if client.get("groups_per_update", 0) < 8:
        errors.append("groups_per_update must be >= 8 for the GPU gate")
    if client.get("max_groups_per_update", 0) < client.get("groups_per_update", 0):
        errors.append("max_groups_per_update must cover groups_per_update")
    if client.get("adaptive_universe", 0) < client.get("max_groups_per_update", 0):
        errors.append("adaptive_universe must cover one distinct-seed update")
    if client.get("iters") != 2:
        errors.append("GPU gate must perform exactly two logical update RPCs")
    if trainer.get("update_epochs", 0) < 2:
        errors.append("update_epochs must be >= 2")
    if trainer.get("sampler") != "pirl" or client.get("joint_logprob") is not False:
        errors.append("latest piRL elementwise non-joint ratio path is required")
    if trainer.get("eta") != client.get("eta"):
        errors.append("trainer/client eta must match")
    if not client.get("skip_eval") or client.get("eval_n") or client.get("heldout_n"):
        errors.append("CPU plan is a GPU-update gate only; eval must stay disabled")
    if errors:
        raise ValueError("; ".join(errors))
    server_source = (Path(__file__).resolve().parent.parent / "src" /
                     "grpo_trainer_server.py").read_text()
    runner = "bash scripts/run-train.sh"
    trainer_args = [
        "--sampler", trainer["sampler"], "--optimizer", trainer["optimizer"],
        "--lr", str(trainer["lr"]), "--weight-decay", str(trainer["weight_decay"]),
        "--train-mode", trainer["train_mode"], "--adapter-skills", trainer["adapter_skills"],
        "--lora-targets", trainer["lora_targets"], "--eta", str(trainer["eta"]),
        "--grad-clip", str(trainer["grad_clip"]),
        "--update-epochs", str(trainer["update_epochs"]),
        "--target-kl", str(trainer["target_kl"]), "--clip", str(trainer["clip"]),
        "--ratio-max", str(trainer["ratio_max"]), "--adv-clip", str(trainer["adv_clip"]),
    ]
    client_args = [
        "--train-skill", client["train_skill"], "--group", str(client["group"]),
        "--groups-per-update", str(client["groups_per_update"]),
        "--min-trainable-chunks", str(client["min_trainable_chunks"]),
        "--max-groups-per-update", str(client["max_groups_per_update"]),
        "--iters", str(client["iters"]), "--adaptive-band",
        *map(str, client["adaptive_band"]), "--adaptive-universe", str(client["adaptive_universe"]),
        "--adaptive-explore-frac", str(client["adaptive_explore_frac"]),
        "--adaptive-ema", str(client["adaptive_ema"]),
        "--adaptive-avoid-recent", str(client["adaptive_avoid_recent"]),
        "--eta", str(client["eta"]), "--reward-variant", client["reward_variant"],
        "--hold-steps", str(client["hold_steps"]), "--update-epochs",
        str(trainer["update_epochs"]), "--target-kl", str(trainer["target_kl"]),
        "--clip", str(trainer["clip"]), "--ratio-max", str(trainer["ratio_max"]),
        "--adv-clip", str(trainer["adv_clip"]), "--eval-n", str(client["eval_n"]),
        "--heldout-n", str(client["heldout_n"]), "--skip-eval",
    ]
    return {
        "validation": "pass",
        "gpu_gate": "pending GPU gate",
        "run": run,
        "source_audit": audit_nonjoint_source(server_source),
        "trainer": trainer,
        "client": {**client, "sampler": trainer["sampler"],
                   "update_epochs": trainer["update_epochs"]},
        "expected": {
            "logical_update_rpcs": client["iters"],
            "optimizer_epochs_per_update": trainer["update_epochs"],
            "minimum_groups_per_update": client["groups_per_update"],
            "trajectories_per_update": client["group"] * client["groups_per_update"],
            "minimum_trainable_chunks_per_update": client["min_trainable_chunks"],
            "checkpoint": f"results/{run}/grpo_trained.pt",
            "progress": f"results/{run}/group_progress.jsonl",
        },
        "commands": {
            "dry_run": f"python scripts/multigroup_production.py --config configs/multigroup_grasp_v1.json --run {run} --dry-run",
            "trainer": f"{runner} trainer-start {shlex.join(trainer_args)}",
            "client": f"{runner} train {shlex.quote(run)} {shlex.join(client_args)}",
            "gpu_execution": f"{runner} multigroup-gpu-gate {shlex.quote(run)}",
        },
        "argv": {"trainer": trainer_args, "client": client_args},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--runner")
    parser.add_argument("--out")
    parser.add_argument("--results-root")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    plan = validate(config, args.run)
    text = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text, end="")
    if args.execute:
        if not args.runner:
            raise ValueError("--runner is required with --execute")
        runner = ["bash", args.runner]
        results_root = (Path(args.results_root) if args.results_root else
                        Path(__file__).resolve().parent.parent / "results")
        run_dir = results_root / args.run
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(runner + ["trainer-start"] + plan["argv"]["trainer"], check=True)
            subprocess.run(runner + ["train", args.run] + plan["argv"]["client"], check=True)
            (run_dir / "DONE").write_text(json.dumps({
                "status": "done", "run": args.run,
            }, sort_keys=True) + "\n")
        except Exception as exc:
            (run_dir / "FAILED").write_text(json.dumps({
                "status": "failed", "run": args.run,
                "error": f"{type(exc).__name__}: {exc}",
            }, sort_keys=True) + "\n")
            raise
        finally:
            subprocess.run(runner + ["trainer-stop"], check=False)


if __name__ == "__main__":
    main()
