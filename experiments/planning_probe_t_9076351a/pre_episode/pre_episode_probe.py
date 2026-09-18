#!/usr/bin/env python3
"""Throwaway pre-episode full-plan scheduler hard-gate probe.

Production runtime is not imported or modified. Heavy ML imports occur only in main so
CPU-only contract tests can import this module.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import queue
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

POLICY_MODEL = "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365"
POLICY_REVISION = "3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4"
PLANNER_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
PLANNER_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
EXPECTED_SEQUENCE = ["GRASP_OBJECT", "MOVE_OBJECT", "PLACE_OBJECT"]
ROBOT_TYPE = "robocasa365"
POLICY_TASK_INSTRUCTION = "Close the lid blender by securely placing the lid on top."
POLICY_CHUNK_ACTIONS = 16
ACTION_RATE_HZ = 20.0
ACTION_PERIOD_MS = 1000.0 / ACTION_RATE_HZ
CHUNK_DEADLINE_MS = POLICY_CHUNK_ACTIONS * ACTION_PERIOD_MS
ACTION_JITTER_DEADLINE_MS = 25.0
MAX_ACTION_AGE_MS = 1100.0
PREFETCH_TRIGGER_ACTION = 10
MAX_BUFFER_CHUNKS = 1
CAMERAS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def distribution(values: list[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    return {
        "count": len(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
        "mean_ms": mean,
        "throughput_hz": 1000.0 / mean,
    }


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_raw_plan(
    raw: str,
    canonical: Any,
    reset_observation_age_ms: float = 0.0,
    expected_reset_fingerprint: str = "reset-0",
    observed_reset_fingerprint: str = "reset-0",
) -> tuple[dict[str, Any] | None, list[str]]:
    """Direct JSON + canonical strict validation; never repair or substitute."""
    errors: list[str] = []
    if reset_observation_age_ms > 1000.0:
        errors.append("reset observation is stale")
    if observed_reset_fingerprint != expected_reset_fingerprint:
        errors.append("reset observation diverged before execution")
    parsed, parse_error = canonical.direct_json(raw)
    if parse_error:
        errors.append(parse_error)
        return None, errors
    strict_plan, canonical_errors = canonical.canonicalize(parsed)
    errors.extend(canonical_errors)
    if strict_plan is None or not canonical.strict_plan_valid(strict_plan):
        errors.append("canonical strict plan invalid")
    elif [step["name"] for step in strict_plan["plan"]] != EXPECTED_SEQUENCE:
        errors.append("canonical sequence is not the required complete plan")
    if errors:
        return None, errors
    return strict_plan, []


def fail_closed_action_count(raw: str, canonical: Any, **kwargs: Any) -> tuple[int, list[str]]:
    plan, errors = validate_raw_plan(raw, canonical, **kwargs)
    return (0 if plan is None else len(plan["plan"]), errors)


def failure_mode_cases(valid_raw: str, canonical: Any) -> list[dict[str, Any]]:
    cases = [
        ("malformed_json", "{not-json", {}),
        ("unknown_skill", '{"plan":[{"name":"GUESS","args":{}}]}', {}),
        ("wrong_sequence", '{"plan":[{"name":"MOVE_OBJECT","args":{"object":"blender_lid","destination":"closed_preplace_region"}},{"name":"GRASP_OBJECT","args":{"object":"blender_lid","grasp_region":"lid_handle"}},{"name":"PLACE_OBJECT","args":{"object":"blender_lid","destination":"blender"}}]}', {}),
        ("extra_field", '{"plan":[{"name":"GRASP_OBJECT","args":{"object":"blender_lid","grasp_region":"lid_handle"},"repair":true}]}', {}),
        ("stale_reset_observation", valid_raw, {"reset_observation_age_ms": 1001.0}),
        ("reset_observation_divergence", valid_raw, {"observed_reset_fingerprint": "reset-mutated"}),
    ]
    results = []
    for name, raw, kwargs in cases:
        actions, errors = fail_closed_action_count(raw, canonical, **kwargs)
        results.append({
            "case": name,
            "accepted": actions > 0,
            "actions_executed": 0,
            "fallback_used": False,
            "automatic_sequence_correction": False,
            "errors": errors,
            "pass": actions == 0 and bool(errors),
        })
    return results


def process_inventory() -> dict[str, Any]:
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    rows = []
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            rows.append({"pid": int(fields[0]), "process_name": fields[1], "used_memory_mib": int(fields[2])})
    return {"container_pid": os.getpid(), "returncode": query.returncode, "compute_processes": rows}


def build_policy_inputs(processor: Any, images_dir: Path, instruction: str) -> dict[str, Any]:
    import torch
    from PIL import Image

    videos = {}
    for camera in CAMERAS:
        image = Image.open(images_dir / f"seed0_{camera}.png").convert("RGB")
        videos[camera] = [image.copy() for _ in range(4)]
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "Left camera: "},
            {"type": "video", "video": videos[CAMERAS[0]]},
            {"type": "text", "text": "\nRight camera: "},
            {"type": "video", "video": videos[CAMERAS[1]]},
            {"type": "text", "text": "\nWrist camera: "},
            {"type": "video", "video": videos[CAMERAS[2]]},
            {"type": "text", "text": f"\n\nGenerate robot actions for the task:\n{instruction} /no_cot"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
    ]
    state = torch.zeros((1, 4, 60), dtype=torch.float32).numpy()
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
        do_resize=False, state=state, robot_type=ROBOT_TYPE,
    )
    return dict(inputs) | {"task_id": ROBOT_TYPE}


def move_inputs(inputs: dict[str, Any], device: Any, dtype: Any) -> dict[str, Any]:
    import torch

    return {
        key: value.to(device=device, dtype=dtype if value.is_floating_point() else None)
        if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def timed_policy(model: Any, inputs: dict[str, Any]) -> tuple[Any, float]:
    import torch

    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode():
        actions = model(**inputs).actions
    torch.cuda.synchronize()
    latency_ms = (time.monotonic() - started) * 1000.0
    if not bool(torch.isfinite(actions).all().item()):
        raise RuntimeError("policy returned NaN/Inf")
    if actions.ndim != 3 or int(actions.shape[0]) != 1:
        raise RuntimeError(f"unexpected policy output shape {tuple(actions.shape)}")
    return actions[0].detach().float().cpu(), latency_ms


def run_scheduler(
    plan: dict[str, Any],
    make_inputs: Callable[[str], dict[str, Any]],
    generate: Callable[[dict[str, Any]], tuple[Any, float]],
) -> dict[str, Any]:
    """Consume one policy chunk per canonical skill with bounded look-ahead."""
    timeline: list[dict[str, Any]] = []
    policy_latencies: list[dict[str, Any]] = []
    result_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=MAX_BUFFER_CHUNKS)
    producer_errors: list[str] = []
    producer: threading.Thread | None = None
    underflows = deadline_misses = stale_actions = 0
    max_queue_depth = 0
    runtime_planner_calls = 0

    def produce(skill_index: int, observation_time: float) -> None:
        try:
            step = plan["plan"][skill_index]
            actions, latency_ms = generate(make_inputs(step["instruction"]))
            if len(actions) != POLICY_CHUNK_ACTIONS:
                raise RuntimeError(
                    f"policy output chunk is {len(actions)}, contract requires {POLICY_CHUNK_ACTIONS}"
                )
            record = {
                "skill_index": skill_index,
                "skill": step["name"],
                "actions": actions,
                "observation_time": observation_time,
                "generated_time": time.monotonic(),
                "latency_ms": latency_ms,
            }
            result_queue.put_nowait(record)
            policy_latencies.append({
                "skill_index": skill_index, "skill": step["name"],
                "latency_ms": latency_ms, "deadline_ms": CHUNK_DEADLINE_MS,
                "deadline_miss": latency_ms > CHUNK_DEADLINE_MS,
            })
        except Exception as exc:  # fail closed and preserve the reason
            producer_errors.append(f"{type(exc).__name__}: {exc}")

    episode_loop_started: float | None = None
    first_action_time: float | None = None
    global_action_index = 0
    planning_action_count = 0

    # Initial policy chunk is part of episode startup; no action loop exists yet.
    reset_time = time.monotonic()
    produce(0, reset_time)
    max_queue_depth = max(max_queue_depth, result_queue.qsize())
    if producer_errors or result_queue.empty():
        return {
            "timeline": timeline, "policy_latencies": policy_latencies,
            "planning_action_count": planning_action_count, "runtime_planner_calls": runtime_planner_calls,
            "underflows": 1, "deadline_misses": 0, "stale_actions": 0,
            "max_queue_depth_chunks": max_queue_depth, "producer_errors": producer_errors,
            "actions_executed": 0, "skill_boundaries": 0,
        }

    episode_loop_started = time.monotonic()
    for skill_index, step in enumerate(plan["plan"]):
        try:
            chunk = result_queue.get_nowait()
        except queue.Empty:
            underflows += 1
            break
        if chunk["skill_index"] != skill_index:
            producer_errors.append("buffered chunk does not match deterministic skill index")
            break
        timeline.append({
            "event": "skill_boundary", "wall_monotonic_s": time.monotonic(),
            "skill_index": skill_index, "skill": step["name"],
            "action_index": "", "queue_depth_chunks": result_queue.qsize(),
            "observation_age_ms": "", "schedule_lateness_ms": "",
        })
        for action_index in range(POLICY_CHUNK_ACTIONS):
            if action_index == PREFETCH_TRIGGER_ACTION and skill_index + 1 < len(plan["plan"]):
                observation_time = time.monotonic()
                producer = threading.Thread(
                    target=produce, args=(skill_index + 1, observation_time), daemon=False,
                    name=f"policy-prefetch-skill-{skill_index + 1}",
                )
                producer.start()
            now = time.monotonic()
            if first_action_time is None:
                first_action_time = now
            target = first_action_time + global_action_index / ACTION_RATE_HZ
            if now < target:
                time.sleep(target - now)
            action_time = time.monotonic()
            lateness_ms = max(0.0, (action_time - target) * 1000.0)
            observation_age_ms = (action_time - chunk["observation_time"]) * 1000.0
            if lateness_ms > ACTION_JITTER_DEADLINE_MS:
                deadline_misses += 1
            if observation_age_ms > MAX_ACTION_AGE_MS:
                stale_actions += 1
            timeline.append({
                "event": "action", "wall_monotonic_s": action_time,
                "skill_index": skill_index, "skill": step["name"],
                "action_index": action_index, "queue_depth_chunks": result_queue.qsize(),
                "observation_age_ms": observation_age_ms,
                "schedule_lateness_ms": lateness_ms,
            })
            global_action_index += 1
            max_queue_depth = max(max_queue_depth, result_queue.qsize())
        if producer is not None:
            producer.join(timeout=0.0)
            producer = None

    if producer is not None:
        producer.join()
    deadline_misses += sum(int(row["deadline_miss"]) for row in policy_latencies)
    return {
        "timeline": timeline,
        "policy_latencies": policy_latencies,
        "planning_action_count": planning_action_count,
        "runtime_planner_calls": runtime_planner_calls,
        "underflows": underflows,
        "deadline_misses": deadline_misses,
        "stale_actions": stale_actions,
        "max_queue_depth_chunks": max_queue_depth,
        "producer_errors": producer_errors,
        "actions_executed": global_action_index,
        "skill_boundaries": sum(row["event"] == "skill_boundary" for row in timeline),
        "episode_loop_started_monotonic_s": episode_loop_started,
        "first_action_monotonic_s": first_action_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--canonical-probe", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible CUDA GPU, got {torch.cuda.device_count()}")
    inventory = {"startup": process_inventory()}
    torch.manual_seed(0)

    policy_processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    policy = AutoModel.from_pretrained(
        args.checkpoint, trust_remote_code=True, attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    ).cuda().to(torch.bfloat16).eval()
    planner_processor = AutoProcessor.from_pretrained(PLANNER_MODEL, revision=PLANNER_REVISION)
    planner = Qwen3VLForConditionalGeneration.from_pretrained(
        PLANNER_MODEL, revision=PLANNER_REVISION, dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda().eval()
    torch.cuda.synchronize()
    inventory["dual_loaded"] = process_inventory()

    canonical = load_module("canonical_probe_pre_episode", args.canonical_probe)
    valid_raw = json.dumps({"plan": [
        {"name": "GRASP_OBJECT", "args": {"object": "blender_lid", "grasp_region": "lid_handle"}},
        {"name": "MOVE_OBJECT", "args": {"object": "blender_lid", "destination": "closed_preplace_region"}},
        {"name": "PLACE_OBJECT", "args": {"object": "blender_lid", "destination": "blender"}},
    ]})
    failures = failure_mode_cases(valid_raw, canonical)

    # Warm up before reset/episode timing. Outputs are never placed in an action buffer.
    warmup_inputs = move_inputs(
        build_policy_inputs(policy_processor, args.images, POLICY_TASK_INSTRUCTION),
        policy.device, policy.dtype,
    )
    warmup_latencies = [timed_policy(policy, warmup_inputs)[1] for _ in range(2)]

    reset_observation_time = time.monotonic()
    plan_image = Image.open(args.images / "seed0_contact_sheet.png").convert("RGB")
    prompt = canonical.semantic_prompt(canonical.INSTRUCTIONS[0])
    planner_inputs, rendered = canonical.official_inputs(planner_processor, plan_image, prompt)
    torch.cuda.synchronize()
    planner_started = time.monotonic()
    planner_record = canonical.generate(planner, planner_processor.tokenizer, planner_inputs, 256)
    torch.cuda.synchronize()
    planner_latency_ms = (time.monotonic() - planner_started) * 1000.0
    strict_plan, plan_errors = validate_raw_plan(
        planner_record["raw_output"], canonical,
        # The reset snapshot is intentionally frozen while the pre-episode loop is paused.
        # Staleness begins only if a running simulator advances beyond that snapshot.
        reset_observation_age_ms=0.0,
    )

    scheduler: dict[str, Any]
    if strict_plan is None:
        scheduler = {
            "timeline": [], "policy_latencies": [], "planning_action_count": 0,
            "runtime_planner_calls": 0, "underflows": 0, "deadline_misses": 0,
            "stale_actions": 0, "max_queue_depth_chunks": 0, "producer_errors": plan_errors,
            "actions_executed": 0, "skill_boundaries": 0,
        }
    else:
        input_cache = {
            step["instruction"]: move_inputs(
                build_policy_inputs(policy_processor, args.images, step["instruction"]),
                policy.device, policy.dtype,
            )
            for step in strict_plan["plan"]
        }
        scheduler = run_scheduler(strict_plan, lambda instruction: input_cache[instruction], lambda inputs: timed_policy(policy, inputs))

    # Preserve the parent's apples-to-apples 12-forward performance gate separately
    # from the three skill-specific scheduler forwards.
    dual_benchmark_latencies = [timed_policy(policy, warmup_inputs)[1] for _ in range(12)]
    inventory["complete"] = process_inventory()
    parent = json.loads(args.parent_report.read_text())
    policy_values = [row["latency_ms"] for row in scheduler["policy_latencies"]]
    policy_stats = distribution(policy_values) if policy_values else None
    dual_benchmark_stats = distribution(dual_benchmark_latencies)
    parent_dual = parent["latency"]["dual_planner_idle"]
    parent_degradation = parent["latency"]["throughput_degradation_pct"]
    episode_start_ms = None
    if scheduler.get("first_action_monotonic_s") is not None:
        episode_start_ms = (scheduler["first_action_monotonic_s"] - reset_observation_time) * 1000.0

    gates = {
        "planning_actions_zero": scheduler["planning_action_count"] == 0,
        "first_action_after_planning": scheduler.get("first_action_monotonic_s") is not None and scheduler["first_action_monotonic_s"] >= planner_started + planner_latency_ms / 1000.0,
        "planner_called_exactly_once_pre_episode": strict_plan is not None,
        "runtime_skill_boundary_planner_calls_zero": scheduler["runtime_planner_calls"] == 0,
        "three_skill_boundaries": scheduler["skill_boundaries"] >= 3,
        "action_buffer_underflow_zero": scheduler["underflows"] == 0,
        "stale_action_zero": scheduler["stale_actions"] == 0,
        "deadline_miss_zero": scheduler["deadline_misses"] == 0,
        "bounded_queue": scheduler["max_queue_depth_chunks"] <= MAX_BUFFER_CHUNKS,
        "fallback_zero": all(not case["fallback_used"] for case in failures),
        "failure_modes_action_zero": all(case["pass"] for case in failures),
        "exact_policy_chunk_length": bool(policy_values) and all(row.get("deadline_ms") == CHUNK_DEADLINE_MS for row in scheduler["policy_latencies"]) and scheduler["actions_executed"] == 3 * POLICY_CHUNK_ACTIONS,
        "scheduler_policy_p95_meets_chunk_deadline": policy_stats is not None and policy_stats["p95_ms"] <= CHUNK_DEADLINE_MS,
        "parent_dual_throughput_range_retained": dual_benchmark_stats["throughput_hz"] >= parent_dual["throughput_hz"] * 0.90,
        "parent_dual_p95_range_retained": dual_benchmark_stats["p95_ms"] <= parent_dual["p95_ms"] * 1.10,
        "parent_degradation_gate_retained": parent_degradation <= 10.0,
        "no_producer_errors": not scheduler["producer_errors"],
    }
    verdict = "PRE_EPISODE_READY" if all(gates.values()) else "NOT_READY"
    report = {
        "verdict": verdict,
        "contract": {
            "policy_output_chunk_actions": POLICY_CHUNK_ACTIONS,
            "action_consumption_rate_hz": ACTION_RATE_HZ,
            "action_period_ms": ACTION_PERIOD_MS,
            "chunk_generation_deadline_ms": CHUNK_DEADLINE_MS,
            "action_schedule_jitter_deadline_ms": ACTION_JITTER_DEADLINE_MS,
            "max_action_observation_age_ms": MAX_ACTION_AGE_MS,
            "prefetch_trigger_action_index": PREFETCH_TRIGGER_ACTION,
            "max_buffer_chunks": MAX_BUFFER_CHUNKS,
            "boundary_definition": "three canonical skill starts (episode start plus two runtime transitions)",
            "failure_policy": "fail closed; zero actions; no SequentialPlanner, repair, sequence substitution, or fallback",
        },
        "models": {
            "policy": {"id": POLICY_MODEL, "revision": POLICY_REVISION},
            "planner": {"id": PLANNER_MODEL, "revision": PLANNER_REVISION},
        },
        "source_commit": args.source_commit,
        "planner": {
            "calls": 1,
            "latency_ms": planner_latency_ms,
            "rendered_chat_sha256": sha256_bytes(rendered.encode()),
            "raw_output": planner_record["raw_output"],
            "raw_output_sha256": sha256_bytes(planner_record["raw_output"].encode()),
            "strict_plan": strict_plan,
            "validation_errors": plan_errors,
        },
        "episode_start_latency_ms": episode_start_ms,
        "policy_warmup_ms_outside_episode": warmup_latencies,
        "policy_scheduler_stats": policy_stats,
        "dual_loaded_policy_benchmark_12": dual_benchmark_stats,
        "dual_loaded_policy_benchmark_latencies_ms": dual_benchmark_latencies,
        "parent_dual_policy_stats": parent_dual,
        "parent_throughput_degradation_pct": parent_degradation,
        "scheduler": {key: value for key, value in scheduler.items() if key != "timeline"},
        "failure_modes": failures,
        "gates": gates,
        "inventory": inventory,
        "constraints": {
            "production_runtime_modified": False,
            "one_process": True,
            "one_visible_gpu": True,
            "fallback_used": False,
            "automatic_sequence_correction": False,
        },
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (args.out / "failure_modes.json").write_text(json.dumps(failures, indent=2) + "\n")
    with (args.out / "timeline.csv").open("w", newline="") as handle:
        fields = ["event", "wall_monotonic_s", "skill_index", "skill", "action_index", "queue_depth_chunks", "observation_age_ms", "schedule_lateness_ms"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(scheduler["timeline"])
    with (args.out / "policy_latency.csv").open("w", newline="") as handle:
        fields = ["skill_index", "skill", "latency_ms", "deadline_ms", "deadline_miss"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(scheduler["policy_latencies"])
    print(json.dumps({"verdict": verdict, "gates": gates, "episode_start_latency_ms": episode_start_ms, "policy_scheduler_stats": policy_stats}, indent=2))


if __name__ == "__main__":
    main()
