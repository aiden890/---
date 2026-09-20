#!/usr/bin/env python3
"""Durable two-arm fresh-base GRASP HPO driver.

The driver is intentionally sequential at the learner: each arm gets fresh resident
actor/learner processes, eight accepted updates, and same-process paired base/trained
held-out evaluation. It never loads the completed v100 lineage.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import distributed_grasp_train as legacy  # noqa: E402

STATE = Path.home() / ".hermes/state/grasp_fast_hpo_after_v100.json"
SPARK = legacy.SPARK
LEARNER = legacy.LEARNER
ACTOR = legacy.ACTOR_CONTAINER
TRAINER = legacy.LEARNER_CONTAINER
SPARK_ROOT = legacy.SPARK_ROOT
LEARNER_ROOT = legacy.LEARNER_ROOT
IMAGE_ACTOR = "rlinf-mibot:spark-a4ee4562"
IMAGE_TRAINER = "xiaomi-cu121:t_9f03a613"
ARMS = (("A", 2e-5), ("B", 5e-5))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def ssh(host: str, command: str, *, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, command], check=True,
                          text=True, capture_output=capture)


def wait_ready(host: str, container: str, timeout: int = 1200) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = ssh(host, f"docker inspect -f '{{{{.State.Running}}}}' {container} 2>/dev/null || true",
                     capture=True).stdout.strip()
        logs = ssh(host, f"docker logs {container} 2>&1 || true", capture=True).stdout
        if "GRPO trainer server on" in logs:
            return
        if status and status != "true":
            raise RuntimeError(f"{host}:{container} exited before ready: {logs[-4000:]}")
        time.sleep(5)
    raise TimeoutError(f"server readiness timeout: {host}:{container}")


def start_fresh_servers(lr: float, source_commit: str) -> dict:
    # Duplicate prevention is checked again immediately before destructive transition.
    for host in (SPARK, LEARNER):
        live = ssh(host, "docker ps --format '{{.Names}}' --filter name=rlinf-mibot-collector-",
                   capture=True).stdout.strip()
        if live:
            raise RuntimeError(f"live collector on {host}: {live}")
    consumer = ssh(LEARNER,
                   "pgrep -af '[s]treaming_consume.py|[a]sync_distributed_grasp_train.py' || true",
                   capture=True).stdout.strip()
    if consumer:
        raise RuntimeError(f"learner still owned by another consumer/coordinator: {consumer}")
    ssh(SPARK, f"docker rm -f {ACTOR} >/dev/null 2>&1 || true")
    ssh(LEARNER, f"docker rm -f {TRAINER} >/dev/null 2>&1 || true")
    common = (
        "python3 /train/src/grpo_trainer_server.py --model /checkpoint --host 127.0.0.1 "
        "--port 10088 --sampler pirl --eta 0.1 --optimizer adamw "
        f"--lr {lr:.17g} --weight-decay 0.01 --grad-clip 1.0 --clip 0.2 --kl-coef 0 "
        "--update-epochs 1 --adapter-skills grasp,move_holding,place "
        "--lora-targets all_linear --rank 16 --alpha 32"
    )
    actor_command = (
        f"docker run -d --name {ACTOR} --gpus all --network host --shm-size=16g "
        f"-e RLINF_SOURCE_COMMIT={source_commit} -e PYTHONPATH=/integration/src:/train/src:/rl_env/src "
        "-e HF_HOME=/cache/hf -e HF_HUB_OFFLINE=1 "
        f"-v {SPARK_ROOT}/checkpoint:/checkpoint:ro -v {SPARK_ROOT}/results:/results "
        f"-v {SPARK_ROOT}/cache:/cache -v {SPARK_ROOT}/source-current/integrations/rlinf_mibot:/integration:ro "
        f"-v {SPARK_ROOT}/source-current/rl-train-t_3ed65912:/train:ro "
        f"-v {SPARK_ROOT}/source-current/rl-env-t_4f3f2b20:/rl_env:ro "
        f"{IMAGE_ACTOR} {common}"
    )
    trainer_command = (
        f"docker run -d --name {TRAINER} --gpus all --network host --shm-size=16g "
        f"-e RLINF_SOURCE_COMMIT={source_commit} -e PYTHONPATH=/integration/src:/train/src:/rl_env/src "
        "-e PYTHONUNBUFFERED=1 "
        "-v /home/guest/robocasa-docker-t_9f03a613/checkpoint:/checkpoint:ro "
        f"-v {LEARNER_ROOT}/integration:/integration:ro -v {LEARNER_ROOT}/train:/train:ro "
        f"-v {LEARNER_ROOT}/env:/rl_env:ro -v {LEARNER_ROOT}/results:/results "
        f"{IMAGE_TRAINER} {common} --no-grad-checkpoint --microbatch-size 4 --post-diag-chunks 128"
    )
    ssh(SPARK, actor_command)
    ssh(LEARNER, trainer_command)
    wait_ready(SPARK, ACTOR)
    wait_ready(LEARNER, TRAINER)
    actor = legacy.rpc(SPARK, ACTOR, {"op": "metrics"})
    learner = legacy.rpc(LEARNER, TRAINER, {"op": "metrics"})
    actor_cfg = legacy.rpc(SPARK, ACTOR, {"op": "config", "code_rev": source_commit})
    learner_cfg = legacy.rpc(LEARNER, TRAINER, {"op": "config", "code_rev": source_commit})
    if int(actor["policy_version"]) != 0 or int(learner["policy_version"]) != 0:
        raise RuntimeError(f"fresh restart did not start at policy v0: {actor}, {learner}")
    if actor["policy_hash"] != learner["policy_hash"]:
        raise RuntimeError("fresh actor/learner LoRA hashes differ")
    for cfg in (actor_cfg["config"], learner_cfg["config"]):
        expected = {"optimizer": "adamw", "lr": lr, "weight_decay": 0.01,
                    "grad_clip": 1.0, "clip": 0.2, "kl_coef": 0.0,
                    "update_epochs": 1, "sampler": "pirl", "eta": 0.1,
                    "lora_targets": "all_linear", "rank": 16, "alpha": 32}
        mismatch = {k: (cfg.get(k), v) for k, v in expected.items() if cfg.get(k) != v}
        if mismatch:
            raise RuntimeError(f"fresh server contract mismatch: {mismatch}")
    return {"actor_metrics": actor, "learner_metrics": learner,
            "actor_config": actor_cfg, "learner_config": learner_cfg,
            "initialization": "fresh_restart_from_original_base_no_checkpoint_load"}


def run_eval(run_id: str, policy_mode: str, source_commit: str) -> list[dict]:
    command = (
        f"RLINF_CHECKPOINT={SPARK_ROOT}/checkpoint RLINF_ASSETS={SPARK_ROOT}/assets "
        f"RLINF_RESULTS={SPARK_ROOT}/results RLINF_CACHE={SPARK_ROOT}/cache "
        f"RLINF_IMAGE={IMAGE_ACTOR} RLINF_WORKERS=4 RLINF_NODE_RANKS=0,0,0,0 "
        f"RLINF_SOURCE_COMMIT={source_commit} bash {legacy.RUNNER} grid-parallel {run_id} "
        "sampler.noise_level=0.0 rollout.groups=100 rollout.group_size=1 "
        "rollout.eval_only=true rollout.save_video=false "
        f"rollout.policy_mode={policy_mode} env.horizon=208 reward.variant=grasp_binary "
        f"seed.base=910000 runtime.max_in_flight=4 run.source_commit={source_commit}"
    )
    ssh(SPARK, command)
    text = ssh(SPARK, f"docker exec {ACTOR} cat /results/{run_id}/episodes.jsonl",
               capture=True).stdout
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    expected_seeds = set(range(910000, 910100))
    seeds = [int(row["seed"]) for row in rows]
    if len(rows) != 100 or set(seeds) != expected_seeds or len(seeds) != len(set(seeds)):
        raise RuntimeError(f"held-out coverage mismatch for {run_id}")
    identities = {(int(row["trainer_memory_before"]["policy_version"]),
                   str(row["trainer_memory_before"]["policy_hash"])) for row in rows}
    if len(identities) != 1:
        raise RuntimeError(f"mixed policy identities in {run_id}: {identities}")
    for row in rows:
        payload = row.get("trainer_payload", {})
        if (payload.get("schema") != "evaluation-only" or
                payload.get("optimizer_update_requested") is not False or
                payload.get("trajectory_ids") != []):
            raise RuntimeError(f"training payload leaked into eval: {row.get('job_id')}")
    return rows


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def summarize_eval(base_rows: list[dict], arm_rows: list[dict], output: Path) -> dict:
    base = {int(row["seed"]): row for row in base_rows}
    arm = {int(row["seed"]): row for row in arm_rows}
    transitions = {"fail_to_fail": 0, "fail_to_success": 0,
                   "success_to_fail": 0, "success_to_success": 0}
    paired = []
    for seed in sorted(base):
        b, a = bool(base[seed]["success"]), bool(arm[seed]["success"])
        key = ("success" if b else "fail") + "_to_" + ("success" if a else "fail")
        transitions[key] += 1
        paired.append({"seed": seed, "base_success": b, "arm_success": a,
                       "base_steps": int(base[seed]["steps"]),
                       "arm_steps": int(arm[seed]["steps"])})
    b_steps = [item["base_steps"] for item in paired if item["base_success"]]
    a_steps = [item["arm_steps"] for item in paired if item["arm_success"]]
    result = {
        "n": 100,
        "base_successes": sum(item["base_success"] for item in paired),
        "arm_successes": sum(item["arm_success"] for item in paired),
        "transitions": transitions,
        "mcnemar_exact_two_sided_p": exact_mcnemar(
            transitions["success_to_fail"], transitions["fail_to_success"]),
        "base_mean_completion_steps_successes": sum(b_steps) / len(b_steps) if b_steps else None,
        "arm_mean_completion_steps_successes": sum(a_steps) / len(a_steps) if a_steps else None,
        "policy_identity": arm_rows[0]["trainer_memory_before"],
        "paired_rows": paired,
    }
    atomic_json(output, result)
    return result


def instability(history: list[dict]) -> tuple:
    worst_kl = worst_clip = 0.0
    min_ess = 1.0
    bad = 0
    for item in history:
        report_path = item["checkpoint"].replace("checkpoint.pt", "consume_smoke.json")
        host_report_path = report_path.replace("/results/", f"{LEARNER_ROOT}/results/", 1)
        text = ssh(LEARNER, f"cat {shlex.quote(host_report_path)}", capture=True).stdout
        update = json.loads(text)["update"]
        bad += int(update.get("n_nonfinite", 0)) + int(update.get("n_dropped", 0))
        worst_kl = max(worst_kl, float(update.get("post_step_mean_kl", 0.0)))
        worst_clip = max(worst_clip, float(update.get("post_step_clip_fraction", 0.0)))
        min_ess = min(min_ess, float(update.get("post_step_ess", 1.0)))
    return bad, max(0.0, worst_kl - 0.02) + max(0.0, worst_clip - 0.30) + max(0.0, 0.95 - min_ess)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    state = {"status": "searching", "source_commit": args.source_commit,
             "source_manifest": str(args.manifest), "run_root": str(args.run_root),
             "driver_pid": os.getpid(), "arms": {}}
    atomic_json(STATE, state)
    args.run_root.mkdir(parents=True, exist_ok=True)
    results = {}
    try:
        for arm, lr in ARMS:
            arm_id = f"grasp-fast-hpo-{arm.lower()}-lr{lr:.0e}-v8"
            arm_dir = args.run_root / arm_id
            arm_dir.mkdir(parents=True, exist_ok=False)
            fresh = start_fresh_servers(lr, args.source_commit)
            atomic_json(arm_dir / "fresh_initialization.json", fresh)
            coordinator_state = arm_dir / "async_state.json"
            env = os.environ.copy()
            env.update({"RLINF_EXPECTED_LR": str(lr), "RLINF_LOCAL_STAGE": str(arm_dir / "stage"),
                        "RLINF_TRAIN_SEED_BASE": "820000" if arm == "A" else "840000"})
            subprocess.run([
                sys.executable, str(HERE / "async_distributed_grasp_train.py"),
                "--updates", "8", "--state", str(coordinator_state),
                "--run-prefix", arm_id,
            ], check=True, env=env)
            coord = json.loads(coordinator_state.read_text())
            if int(coord["accepted_updates"]) != 8 or int(coord["actor"]["version"]) != 8:
                raise RuntimeError(f"arm {arm} did not reach eight accepted updates")
            base_id = f"{arm_id}-heldout-base-100"
            trained_id = f"{arm_id}-heldout-trained-100"
            base_rows = run_eval(base_id, "base", args.source_commit)
            trained_rows = run_eval(trained_id, "trained", args.source_commit)
            evaluation = summarize_eval(base_rows, trained_rows, arm_dir / "paired_eval.json")
            results[arm] = {"arm": arm, "lr": lr, "run_id": arm_id,
                            "coordinator_state": str(coordinator_state),
                            "final_checkpoint": coord["history"][-1]["checkpoint"],
                            "final_policy": coord["actor"],
                            "base_eval_path": f"{SPARK_ROOT}/results/{base_id}",
                            "trained_eval_path": f"{SPARK_ROOT}/results/{trained_id}",
                            "paired_eval_path": str(arm_dir / "paired_eval.json"),
                            "evaluation": evaluation,
                            "instability": instability(coord["history"])}
            state["arms"][arm] = results[arm]
            atomic_json(STATE, state)
        ranked = sorted(results.values(), key=lambda item: (
            -item["evaluation"]["arm_successes"],
            item["evaluation"]["transitions"]["success_to_fail"],
            item["instability"],
            0 if item["arm"] == "A" else 1,
        ))
        selected = ranked[0]
        improved = selected["evaluation"]["arm_successes"] > selected["evaluation"]["base_successes"]
        state.update({"status": "selected", "selected_configuration": selected,
                      "selection_label": "SELECTED_IMPROVED" if improved else "SELECTED_NO_IMPROVEMENT",
                      "completed_at": time.time(), "contract": manifest.get("contract")})
        atomic_json(STATE, state)
        atomic_json(args.run_root / "SELECTED.json", state)
    except Exception as exc:
        state.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                      "failed_at": time.time()})
        atomic_json(STATE, state)
        atomic_json(args.run_root / "FAILED.json", state)
        raise


if __name__ == "__main__":
    main()
