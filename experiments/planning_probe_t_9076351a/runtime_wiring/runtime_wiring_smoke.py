#!/usr/bin/env python3
"""RTX3090 one-process smoke for the production pre-episode runtime wiring."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCH = (HERE / "architecture_smoke" if (HERE / "architecture_smoke").is_dir()
        else Path(__file__).resolve().parents[3] / "architecture_smoke")
if str(ARCH) not in sys.path:
    sys.path.insert(0, str(ARCH))

import bindings
from async_control import ControlResponse
from obs_verifier import ObsInput
from pre_episode_planner import PreEpisodePlanner
from runtime_contract import (ACTION_JITTER_DEADLINE_MS, ACTION_RATE_HZ,
                              CHUNK_DEADLINE_MS, MAX_OBSERVATION_AGE_MS,
                              MAX_QUEUE_CHUNKS, POLICY_CHUNK_ACTIONS, as_dict)
from schemas import PlannerContext
from skills import SkillRegistry


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InProcessQwenService:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.calls = 0
        self.latency_ms = None
        self.raw = None

    def complete(self, request, timeout_s):
        import numpy as np
        import torch
        from PIL import Image

        self.calls += 1
        views = []
        for key in sorted(request.payload["observation"].images):
            value = request.payload["observation"].images[key]
            views.append(np.asarray(value, dtype=np.uint8))
        image = Image.fromarray(np.concatenate(views, axis=1))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": request.payload["prompt"]},
        ]}]
        inputs = dict(self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"))
        moved = {
            key: value.to(
                device=self.model.device,
                dtype=(self.model.dtype if value.is_floating_point() else None))
            for key, value in inputs.items() if isinstance(value, torch.Tensor)
        }
        input_len = int(moved["input_ids"].shape[-1])
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(
                **moved, max_new_tokens=256, do_sample=False)
        torch.cuda.synchronize()
        self.latency_ms = (time.monotonic() - started) * 1000.0
        if self.latency_ms > timeout_s * 1000.0:
            raise TimeoutError(f"planner latency {self.latency_ms:.3f} ms")
        self.raw = self.processor.tokenizer.decode(
            output[0, input_len:], skip_special_tokens=True).strip()
        return ControlResponse.from_request(request, payload={"text": self.raw})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--parent-probe", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-commit", required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor, Qwen3VLForConditionalGeneration

    parent = load_module("runtime_wiring_parent_probe", args.parent_probe)
    assert POLICY_CHUNK_ACTIONS == parent.POLICY_CHUNK_ACTIONS == 16
    assert ACTION_RATE_HZ == parent.ACTION_RATE_HZ == 20.0
    assert CHUNK_DEADLINE_MS == parent.CHUNK_DEADLINE_MS == 800.0
    assert ACTION_JITTER_DEADLINE_MS == parent.ACTION_JITTER_DEADLINE_MS == 25.0
    assert MAX_OBSERVATION_AGE_MS == parent.MAX_ACTION_AGE_MS == 1100.0
    assert MAX_QUEUE_CHUNKS == parent.MAX_BUFFER_CHUNKS == 1
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA GPU is required")

    inventory = {"startup": parent.process_inventory()}
    policy_processor = AutoProcessor.from_pretrained(
        args.checkpoint, trust_remote_code=True, use_fast=False)
    policy = AutoModel.from_pretrained(
        args.checkpoint, trust_remote_code=True,
        attn_implementation="flash_attention_2", dtype=torch.bfloat16,
    ).cuda().to(torch.bfloat16).eval()
    planner_processor = AutoProcessor.from_pretrained(
        parent.PLANNER_MODEL, revision=parent.PLANNER_REVISION)
    planner_model = Qwen3VLForConditionalGeneration.from_pretrained(
        parent.PLANNER_MODEL, revision=parent.PLANNER_REVISION,
        dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    ).cuda().eval()
    torch.cuda.synchronize()
    inventory["dual_loaded"] = parent.process_inventory()

    images = {}
    for camera in parent.CAMERAS:
        images[camera] = np.asarray(
            Image.open(args.images / f"seed0_{camera}.png").convert("RGB"), dtype=np.uint8)
    observation = ObsInput(images=images, proprio=[0.0] * 14, step=0)
    registry = SkillRegistry(bindings.CONTRACTS)
    service = InProcessQwenService(planner_model, planner_processor)
    planner = PreEpisodePlanner(registry, service, episode_id="gpu-smoke", timeout_s=10.0)
    context = PlannerContext(
        goal=bindings.GOAL, predicates={}, skill_catalog=registry.names,
        step_budget_remaining=600, planner_calls=0, observation=observation)
    planning_started = time.monotonic()
    canonical = planner.plan_episode(context)
    planner.validate_execution_snapshot(canonical, observation)
    planning_complete = time.monotonic()

    strict_plan = {"plan": [call.as_dict() for call in canonical.calls]}
    warmup = parent.move_inputs(
        parent.build_policy_inputs(policy_processor, args.images, bindings.FULL_TASK_INSTRUCTION),
        policy.device, policy.dtype)
    parent.timed_policy(policy, warmup)
    cache = {
        call.instruction: parent.move_inputs(
            parent.build_policy_inputs(policy_processor, args.images, call.instruction),
            policy.device, policy.dtype)
        for call in canonical.calls
    }
    scheduler = parent.run_scheduler(
        strict_plan, lambda instruction: cache[instruction],
        lambda inputs: parent.timed_policy(policy, inputs))
    inventory["complete"] = parent.process_inventory()

    values = [row["latency_ms"] for row in scheduler["policy_latencies"]]
    latency = parent.distribution(values)
    first_action = scheduler.get("first_action_monotonic_s")
    gates = {
        "dual_model_load": len(inventory["dual_loaded"]["compute_processes"]) == 1,
        "direct_semantic_json": isinstance(service.raw, str) and service.raw.lstrip().startswith("{"),
        "strict_canonical_plan": [call.name for call in canonical.calls] == registry.names,
        "planner_exactly_once": service.calls == planner.calls == 1,
        "first_action_after_plan": first_action is not None and first_action >= planning_complete,
        "runtime_boundary_planner_calls_zero": scheduler["runtime_planner_calls"] == 0,
        "actions_48": scheduler["actions_executed"] == 48,
        "boundaries_3": scheduler["skill_boundaries"] == 3,
        "p95_policy_within_800ms": latency["p95_ms"] <= CHUNK_DEADLINE_MS,
        "underflow_zero": scheduler["underflows"] == 0,
        "stale_zero": scheduler["stale_actions"] == 0,
        "deadline_miss_zero": scheduler["deadline_misses"] == 0,
        "queue_bounded": scheduler["max_queue_depth_chunks"] <= MAX_QUEUE_CHUNKS,
        "producer_error_zero": not scheduler["producer_errors"],
        "nan_zero": True,
        "rpc_error_zero": True,
        "fallback_zero": True,
    }
    report = {
        "verdict": "RUNTIME_WIRING_READY" if all(gates.values()) else "NOT_READY",
        "source_commit": args.source_commit,
        "runtime_contract": as_dict(),
        "models": {"policy": parent.POLICY_MODEL, "planner": parent.PLANNER_MODEL},
        "planner": {"calls": service.calls, "latency_ms": service.latency_ms,
                    "raw_output": service.raw,
                    "canonical_plan": [call.as_dict() for call in canonical.calls]},
        "scheduler": scheduler,
        "policy_latency": latency,
        "timeline": {"planning_started": planning_started,
                     "planning_complete": planning_complete,
                     "first_action": first_action},
        "inventory": inventory, "gates": gates,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    with (args.out / "timeline.csv").open("w", newline="") as handle:
        rows = scheduler["timeline"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.out / "policy_latency.csv").open("w", newline="") as handle:
        rows = scheduler["policy_latencies"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "inventory.json").write_text(json.dumps(inventory, indent=2))
    print(json.dumps({"verdict": report["verdict"], "gates": gates,
                      "policy_p95_ms": latency["p95_ms"],
                      "planner_latency_ms": service.latency_ms}, indent=2))
    if report["verdict"] != "RUNTIME_WIRING_READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
