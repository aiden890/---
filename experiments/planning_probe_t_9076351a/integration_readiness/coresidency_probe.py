#!/usr/bin/env python3
"""Throwaway single-process Xiaomi policy + Qwen planner co-residency probe."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, Qwen3VLForConditionalGeneration

POLICY_MODEL = "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365"
POLICY_REVISION = "3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4"
PLANNER_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
PLANNER_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
ROBOT_TYPE = "robocasa365"
POLICY_INSTRUCTION = "Close the lid blender by securely placing the lid on top."
CAMERAS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty values")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
        "mean_ms": statistics.fmean(values),
        "throughput_hz": 1000.0 / statistics.fmean(values),
    }


def synchronize() -> None:
    torch.cuda.synchronize()


def gpu_snapshot(stage: str) -> dict[str, Any]:
    synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "stage": stage,
        "monotonic_s": time.monotonic(),
        "allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "driver_free_mib": free_bytes / 2**20,
        "total_mib": total_bytes / 2**20,
    }


def timed_cuda_call(label: str, fn: Callable[[], Any], timeline: list[dict[str, Any]]) -> tuple[Any, float]:
    torch.cuda.reset_peak_memory_stats()
    synchronize()
    started = time.monotonic()
    result = fn()
    synchronize()
    latency_ms = (time.monotonic() - started) * 1000.0
    point = gpu_snapshot(label)
    point["latency_ms"] = latency_ms
    timeline.append(point)
    return result, latency_ms


def process_inventory() -> dict[str, Any]:
    namespace_pids: list[int] = []
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("NSpid:"):
            namespace_pids = [int(value) for value in line.split()[1:]]
            break
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    rows = []
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            rows.append({"pid": int(fields[0]), "process_name": fields[1], "used_memory_mib": int(fields[2])})
    return {
        "container_pid": os.getpid(),
        "namespace_pids_host_to_container": namespace_pids,
        "host_pid": namespace_pids[0] if namespace_pids else os.getpid(),
        "returncode": query.returncode,
        "compute_processes": rows,
    }


def build_policy_inputs(processor: Any, images_dir: Path) -> dict[str, Any]:
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
            {"type": "text", "text": f"\n\nGenerate robot actions for the task:\n{POLICY_INSTRUCTION} /no_cot"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
    ]
    state = torch.zeros((1, 4, 60), dtype=torch.float32).numpy()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        do_resize=False,
        state=state,
        robot_type=ROBOT_TYPE,
    )
    return dict(inputs) | {"task_id": ROBOT_TYPE}


def move_inputs(inputs: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    return {
        key: value.to(device=device, dtype=dtype if value.is_floating_point() else None)
        if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def policy_forward(model: Any, inputs: dict[str, Any]) -> torch.Tensor:
    with torch.inference_mode():
        actions = model(**inputs).actions
    if not bool(torch.isfinite(actions).all().item()):
        raise RuntimeError("policy returned NaN/Inf")
    return actions


def load_canonical_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("canonical_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--canonical-probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy-repeats", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible CUDA GPU, got {torch.cuda.device_count()}")
    torch.manual_seed(0)
    timeline: list[dict[str, Any]] = [gpu_snapshot("startup")]
    errors: list[dict[str, str]] = []
    inventory = {"startup": process_inventory()}

    load_started = time.monotonic()
    policy_processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    policy = AutoModel.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    ).cuda().to(torch.bfloat16).eval()
    synchronize()
    policy_load_ms = (time.monotonic() - load_started) * 1000.0
    timeline.append(gpu_snapshot("policy_loaded"))
    policy_inputs = move_inputs(build_policy_inputs(policy_processor, args.images), policy.device, policy.dtype)

    baseline_warmup = []
    for index in range(args.warmup):
        _, latency = timed_cuda_call(f"baseline_warmup_{index}", lambda: policy_forward(policy, policy_inputs), timeline)
        baseline_warmup.append(latency)
    baseline = []
    for index in range(args.policy_repeats):
        _, latency = timed_cuda_call(f"baseline_{index}", lambda: policy_forward(policy, policy_inputs), timeline)
        baseline.append(latency)

    planner_load_started = time.monotonic()
    planner_processor = AutoProcessor.from_pretrained(PLANNER_MODEL, revision=PLANNER_REVISION)
    planner = Qwen3VLForConditionalGeneration.from_pretrained(
        PLANNER_MODEL,
        revision=PLANNER_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda().eval()
    synchronize()
    planner_load_ms = (time.monotonic() - planner_load_started) * 1000.0
    timeline.append(gpu_snapshot("dual_loaded"))
    inventory["dual_loaded"] = process_inventory()

    dual_warmup = []
    for index in range(args.warmup):
        _, latency = timed_cuda_call(f"dual_warmup_{index}", lambda: policy_forward(policy, policy_inputs), timeline)
        dual_warmup.append(latency)
    dual_idle = []
    for index in range(args.policy_repeats):
        _, latency = timed_cuda_call(f"dual_idle_{index}", lambda: policy_forward(policy, policy_inputs), timeline)
        dual_idle.append(latency)

    canonical = load_canonical_module(args.canonical_probe)
    plan_image = Image.open(args.images / "seed0_contact_sheet.png").convert("RGB")
    prompt = canonical.semantic_prompt(canonical.INSTRUCTIONS[0])
    planner_inputs, rendered = canonical.official_inputs(planner_processor, plan_image, prompt)
    planner_record, planner_latency = timed_cuda_call(
        "planner_generate",
        lambda: canonical.generate(planner, planner_processor.tokenizer, planner_inputs, args.max_new_tokens),
        timeline,
    )
    parsed, parse_error = canonical.direct_json(planner_record["raw_output"])
    validation = canonical.validate_semantic_plan(parsed)
    strict_plan, canonical_errors = canonical.canonicalize(parsed)
    planner_case = {
        "seed": 0,
        "instruction_index": 0,
        "instruction": canonical.INSTRUCTIONS[0],
        "rendered_chat_length": len(rendered),
        "raw_output": planner_record["raw_output"],
        "generated_token_count": planner_record["generated_token_count"],
        "latency_ms": planner_latency,
        "parse_error": parse_error,
        "direct_semantic_json": parsed is not None,
        "validation": validation,
        "canonicalization_errors": canonical_errors,
        "canonical_strict_plan": strict_plan,
        "canonical_strict_plan_valid": canonical.strict_plan_valid(strict_plan),
    }

    _, post_planner_latency = timed_cuda_call(
        "post_planner_policy", lambda: policy_forward(policy, policy_inputs), timeline
    )
    inventory["complete"] = process_inventory()

    baseline_stats = distribution(baseline)
    dual_stats = distribution(dual_idle)
    degradation_pct = (dual_stats["throughput_hz"] / baseline_stats["throughput_hz"] - 1.0) * -100.0
    peak_reserved_mib = max(point["peak_reserved_mib"] for point in timeline)
    total_mib = timeline[0]["total_mib"]
    headroom_mib = total_mib - peak_reserved_mib
    # nvidia-smi reports host PIDs while Docker's default PID namespace exposes only PID 1.
    # The host-side idle gate is checked immediately before launch; exactly one compute row
    # throughout this process therefore identifies this container's single Python process.
    one_process = all(len(item["compute_processes"]) == 1 for item in inventory.values())
    gate = {
        "single_process_single_gpu_serialized": one_process and torch.cuda.device_count() == 1,
        "process_attribution": "host idle before launch; exactly one nvidia-smi compute row throughout; container PID namespace prevents numeric PID equality",
        "zero_oom_nan_rpc_failures": not errors,
        "reserved_headroom_at_least_2_gib": headroom_mib >= 2048.0,
        "policy_throughput_degradation_at_most_10pct": degradation_pct <= 10.0,
        "policy_p95_meets_existing_budget": None,
        "policy_budget_evidence": "No explicit policy-forward/control-loop deadline exists in the inspected production server or rollout scheduler; inference is synchronous and blocks action execution.",
        "planner_boundary_max_latency_ms": planner_latency,
        "action_loop_during_planner": "blocked: one process, serialized CUDA, synchronous call; no queue is created",
        "unbounded_queue_or_backpressure": False,
    }
    if not planner_case["canonical_strict_plan_valid"]:
        verdict = "PLANNER_SFT_REQUIRED"
    elif not gate["reserved_headroom_at_least_2_gib"] or not gate["policy_throughput_degradation_at_most_10pct"]:
        verdict = "SEPARATE_HARDWARE_REQUIRED"
    elif not gate["single_process_single_gpu_serialized"] or not gate["zero_oom_nan_rpc_failures"] or gate["unbounded_queue_or_backpressure"]:
        verdict = "INCONCLUSIVE"
    elif gate["policy_p95_meets_existing_budget"] is None:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "CORESIDENT_READY"

    result = {
        "verdict": verdict,
        "process": {"pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cuda_device_count": torch.cuda.device_count()},
        "gpu": {"name": torch.cuda.get_device_name(0), "total_mib": total_mib},
        "models": {
            "policy": {"id": POLICY_MODEL, "revision": POLICY_REVISION, "path": str(args.checkpoint), "load_ms": policy_load_ms},
            "planner": {"id": PLANNER_MODEL, "revision": PLANNER_REVISION, "load_ms": planner_load_ms},
        },
        "policy_input": {"seed": 0, "history": 4, "state": "zeros(1,4,60)", "instruction": POLICY_INSTRUCTION, "camera_files": [f"seed0_{camera}.png" for camera in CAMERAS]},
        "latency": {
            "baseline_warmup_ms": baseline_warmup,
            "baseline": baseline_stats,
            "dual_warmup_ms": dual_warmup,
            "dual_planner_idle": dual_stats,
            "throughput_degradation_pct": degradation_pct,
            "post_planner_policy_ms": post_planner_latency,
        },
        "memory": {"peak_reserved_mib": peak_reserved_mib, "headroom_mib": headroom_mib},
        "planner_case": planner_case,
        "gate": gate,
        "errors": errors,
        "process_inventory": inventory,
        "timeline": timeline,
    }
    (args.out / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    with (args.out / "latency.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["mode", "index", "latency_ms"])
        for mode, values in (("baseline_warmup", baseline_warmup), ("baseline", baseline), ("dual_warmup", dual_warmup), ("dual_idle", dual_idle)):
            writer.writerows((mode, index, value) for index, value in enumerate(values))
        writer.writerow(("planner", 0, planner_latency))
        writer.writerow(("post_planner_policy", 0, post_planner_latency))
    with (args.out / "vram_timeline.csv").open("w", newline="") as handle:
        fields = sorted({key for point in timeline for key in point})
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(timeline)
    print(json.dumps({"verdict": verdict, "gate": gate, "latency": result["latency"], "memory": result["memory"]}, indent=2))


if __name__ == "__main__":
    main()
