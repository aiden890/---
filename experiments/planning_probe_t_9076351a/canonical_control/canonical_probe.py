#!/usr/bin/env python3
"""Direct semantic-plan probe plus fail-closed deterministic registry materialization."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
INSTRUCTIONS = [
    "Close the lid blender by securely placing the lid on top.",
    "Securely place the blender lid on top to close the blender.",
    "Close the blender by putting its lid securely in place.",
    "Put the lid securely on the blender so that it is closed.",
]
EXPECTED_SEQUENCE = ["GRASP_OBJECT", "MOVE_OBJECT", "PLACE_OBJECT"]
REGISTRY = {
    "GRASP_OBJECT": {
        "description": "Grasp an object at a named grasp region and lift it clear.",
        "args": {"object": ["blender_lid"], "grasp_region": ["lid_handle"]},
        "instruction": "Grasp the blender lid securely at its handle and lift it clear.",
        "contract": "held AND lifted>=0.05m AND stable_for(20) AND no_disallowed_contact",
        "budget": 208,
    },
    "MOVE_OBJECT": {
        "description": "Move a held object to a named destination region.",
        "args": {"object": ["blender_lid"], "destination": ["closed_preplace_region"]},
        "instruction": "Move the held blender lid to the pre-place region above the blender while maintaining a stable grasp and avoiding disallowed contact.",
        "contract": "held AND at_target(xy<=0.06m,z 0.015-0.25m) AND stable_for(3) AND no_disallowed_contact",
        "budget": 288,
    },
    "PLACE_OBJECT": {
        "description": "Place a held object on a support, release, and retreat.",
        "args": {"object": ["blender_lid"], "destination": ["blender"]},
        "instruction": "Place the blender lid on the blender, release it after it is stably supported, then move the gripper clear.",
        "contract": "supported_by AND released AND upright_error<=7deg AND gripper_clear>=0.15m",
        "budget": 96,
    },
}
SEMANTIC_FIELDS = {"name", "args"}
STRICT_FIELDS = {"name", "args", "instruction", "contract", "budget"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def direct_json(raw: str) -> tuple[Any | None, str | None]:
    try:
        return json.loads(raw), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def semantic_prompt(instruction: str) -> str:
    catalog = [
        {"name": name, "description": spec["description"], "allowed_args": spec["args"]}
        for name, spec in REGISTRY.items()
    ]
    payload = {
        "task_instruction": instruction,
        "skill_registry": catalog,
        "output_schema": {"plan": [{"name": "registry skill name", "args": {"required_arg": "exact allowed value"}}]},
    }
    return (
        "Decide the complete robot skill sequence needed for the task visible in the initial camera observation. "
        "Use only the task instruction, image, and supplied registry. Return exactly one JSON object and no prose. "
        "Every step must contain exactly name and args; copy each argument value exactly from allowed_args. "
        "Do not emit instruction, contract, budget, simulator predicates, state labels, or unstated skills.\n"
        + json.dumps(payload, sort_keys=True)
    )


def validate_semantic_plan(data: Any) -> dict[str, Any]:
    result = {
        "semantic_schema_valid": False,
        "name_args_valid": False,
        "sequence": [],
        "exact_sequence": False,
        "first_skill_correct": False,
        "unknown_skill": False,
        "missing_step": True,
        "extra_step": False,
        "duplicate_step": False,
        "errors": [],
    }
    if not isinstance(data, dict) or set(data) != {"plan"} or not isinstance(data.get("plan"), list):
        result["errors"].append("top-level value must be exactly {'plan': list}")
        return result
    plan = data["plan"]
    result["sequence"] = [step.get("name") if isinstance(step, dict) else None for step in plan]
    result["exact_sequence"] = result["sequence"] == EXPECTED_SEQUENCE
    result["first_skill_correct"] = bool(plan) and result["sequence"][0] == EXPECTED_SEQUENCE[0]
    result["unknown_skill"] = any(name not in REGISTRY for name in result["sequence"])
    result["missing_step"] = len(plan) < len(EXPECTED_SEQUENCE)
    result["extra_step"] = len(plan) > len(EXPECTED_SEQUENCE)
    result["duplicate_step"] = len(result["sequence"]) != len(set(map(str, result["sequence"])))
    valid = True
    for index, step in enumerate(plan):
        if not isinstance(step, dict) or set(step) != SEMANTIC_FIELDS:
            result["errors"].append(f"step {index}: fields must be exactly name,args")
            valid = False
            continue
        name, args = step["name"], step["args"]
        if not isinstance(name, str) or name not in REGISTRY:
            result["errors"].append(f"step {index}: unknown skill {name!r}")
            valid = False
            continue
        allowed = REGISTRY[name]["args"]
        if not isinstance(args, dict) or set(args) != set(allowed):
            result["errors"].append(f"step {index}: argument keys do not match registry")
            valid = False
            continue
        for key, value in args.items():
            if not isinstance(value, str) or value not in allowed[key]:
                result["errors"].append(f"step {index}: disallowed {key}={value!r}")
                valid = False
    result["name_args_valid"] = valid
    result["semantic_schema_valid"] = valid
    return result


def canonicalize(data: Any) -> tuple[dict[str, Any] | None, list[str]]:
    validation = validate_semantic_plan(data)
    errors = list(validation["errors"])
    if not validation["semantic_schema_valid"]:
        return None, errors
    if len(data["plan"]) != len(EXPECTED_SEQUENCE):
        errors.append("plan must contain exactly three steps")
    if validation["duplicate_step"]:
        errors.append("duplicate skill names are forbidden")
    if errors:
        return None, errors
    plan = []
    for step in data["plan"]:
        spec = REGISTRY[step["name"]]
        plan.append({
            "name": step["name"],
            "args": dict(step["args"]),
            "instruction": spec["instruction"],
            "contract": spec["contract"],
            "budget": spec["budget"],
        })
    return {"plan": plan}, []


def strict_plan_valid(data: Any) -> bool:
    if not isinstance(data, dict) or set(data) != {"plan"} or not isinstance(data["plan"], list):
        return False
    for step in data["plan"]:
        if not isinstance(step, dict) or set(step) != STRICT_FIELDS:
            return False
        name = step.get("name")
        if name not in REGISTRY:
            return False
        spec = REGISTRY[name]
        if step["args"] != {key: values[0] for key, values in spec["args"].items()}:
            return False
        if step["instruction"] != spec["instruction"] or step["contract"] != spec["contract"]:
            return False
        if step["budget"] != spec["budget"]:
            return False
    return len(data["plan"]) == 3


def official_inputs(processor: Any, image: Any, prompt: str) -> tuple[dict[str, Any], str]:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    features = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    return dict(features), rendered


def generate(model: Any, tokenizer: Any, inputs: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    import torch

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    moved = {
        key: value.to(device=device, dtype=dtype if value.is_floating_point() else None)
        for key, value in inputs.items() if isinstance(value, torch.Tensor)
    }
    input_len = int(moved["input_ids"].shape[-1])
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            **moved, max_new_tokens=max_new_tokens, do_sample=False,
            return_dict_in_generate=True, output_scores=True,
        )
    latency_ms = (time.monotonic() - started) * 1000.0
    token_ids = output.sequences[0, input_len:].detach().cpu().tolist()
    eos = model.generation_config.eos_token_id
    eos_ids = [int(value) for value in eos] if isinstance(eos, (list, tuple)) else [int(eos)]
    first_topk = []
    if output.scores:
        scores = output.scores[0][0].float().cpu()
        values, indices = torch.topk(scores, k=min(10, scores.numel()))
        first_topk = [
            {"token_id": int(token_id), "logit": float(logit),
             "decoded": tokenizer.decode([int(token_id)], skip_special_tokens=False)}
            for logit, token_id in zip(values, indices)
        ]
    return {
        "raw_output": tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
        "raw_output_with_special_tokens": tokenizer.decode(token_ids, skip_special_tokens=False),
        "generated_token_ids": token_ids,
        "generated_token_count": len(token_ids),
        "first_token_topk": first_topk,
        "stop_reason": "eos_token" if token_ids and token_ids[-1] in eos_ids else (
            "max_new_tokens" if len(token_ids) >= max_new_tokens else "generation_ended_other"),
        "eos_token_ids": eos_ids,
        "latency_ms": latency_ms,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "input_token_count": input_len,
    }


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return os.environ.get("SOURCE_COMMIT", "unknown")


def run(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    out = args.out
    raw_dir = out / "raw"
    canonical_dir = out / "canonical"
    raw_dir.mkdir(parents=True, exist_ok=True)
    canonical_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda().eval()
    tokenizer = processor.tokenizer
    cases = []
    for seed in (0, 1, 2):
        image_path = args.images / f"seed{seed}_contact_sheet.png"
        image = Image.open(image_path).convert("RGB")
        for instruction_index, instruction in enumerate(INSTRUCTIONS):
            case_id = f"seed{seed}_instruction{instruction_index}"
            prompt = semantic_prompt(instruction)
            inputs, rendered = official_inputs(processor, image, prompt)
            record = generate(model, tokenizer, inputs, args.max_new_tokens)
            parsed, parse_error = direct_json(record["raw_output"])
            validation = validate_semantic_plan(parsed)
            canonical, canonical_errors = canonicalize(parsed)
            raw_path = raw_dir / f"{case_id}.txt"
            canonical_path = canonical_dir / f"{case_id}.json"
            raw_path.write_text(record["raw_output"], encoding="utf-8")
            if canonical is not None:
                canonical_path.write_text(json.dumps(canonical, indent=2) + "\n", encoding="utf-8")
            record.update(validation)
            record.update({
                "case_id": case_id,
                "seed": seed,
                "instruction_index": instruction_index,
                "instruction": instruction,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "rendered_chat_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "image_path": str(image_path),
                "image_sha256": sha256(image_path),
                "direct_valid_json": parsed is not None,
                "parse_error": parse_error,
                "canonicalization_success": canonical is not None,
                "canonicalization_errors": canonical_errors,
                "canonicalized_plan": canonical,
                "canonical_strict_plan_valid": strict_plan_valid(canonical),
                "raw_path": str(raw_path.relative_to(out)),
                "canonical_path": str(canonical_path.relative_to(out)) if canonical is not None else None,
                "fallback_used": False,
                "parser_repair_used": False,
                "constrained_decoding_used": False,
            })
            cases.append(record)
            print(case_id, json.dumps({
                "json": record["direct_valid_json"], "sequence": record["sequence"],
                "name_args": record["name_args_valid"], "canonical": record["canonicalization_success"],
                "strict": record["canonical_strict_plan_valid"], "latency_ms": record["latency_ms"],
            }), flush=True)
    result = {
        "model": args.model,
        "revision": args.revision,
        "source_commit": git_head(),
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "constraints": {
            "fallback_used": False, "parser_repair_used": False,
            "constrained_decoding_used": False, "production_runtime_modified": False,
        },
        "registry": REGISTRY,
        "instructions": INSTRUCTIONS,
        "cases": cases,
    }
    (out / "cases.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_aggregate(out, result)


def write_aggregate(out: Path, result: dict[str, Any]) -> None:
    cases = result["cases"]
    count = lambda key: sum(bool(case[key]) for case in cases)
    total = len(cases)
    summary = {
        "verdict": "PASS" if all(
            count(key) == total for key in (
                "direct_valid_json", "exact_sequence", "canonical_strict_plan_valid")) else "FAIL/NOT READY",
        "cases": total,
        "direct_semantic_json": count("direct_valid_json"),
        "exact_three_skill_sequence": count("exact_sequence"),
        "model_emitted_name_args_valid": count("name_args_valid"),
        "deterministic_canonicalization_success": count("canonicalization_success"),
        "canonicalized_strict_five_field_plan": count("canonical_strict_plan_valid"),
        "first_skill_correct": count("first_skill_correct"),
        "latency_ms": {
            "min": min(case["latency_ms"] for case in cases),
            "mean": sum(case["latency_ms"] for case in cases) / total,
            "max": max(case["latency_ms"] for case in cases),
        },
        "peak_vram_mib": max(case["peak_vram_mib"] for case in cases),
        "hard_gate": "direct semantic JSON, exact sequence, and canonical strict plan must each be 12/12",
        "constraints": result["constraints"],
        "model": result["model"],
        "revision": result["revision"],
        "source_commit": result["source_commit"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    excluded = {"prompt", "raw_output", "raw_output_with_special_tokens", "first_token_topk", "canonicalized_plan"}
    rows = [{
        key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in case.items() if key not in excluded
    } for case in cases]
    with (out / "cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = f"""# Generic Qwen planner canonical contract 12-case gate

## 판정

**{summary['verdict']}** — direct semantic JSON {summary['direct_semantic_json']}/{total}, exact 3-skill sequence {summary['exact_three_skill_sequence']}/{total}, model-emitted name/args validity {summary['model_emitted_name_args_valid']}/{total}, deterministic canonicalization {summary['deterministic_canonicalization_success']}/{total}, canonicalized strict 5-field plan {summary['canonicalized_strict_five_field_plan']}/{total}, 첫 skill {summary['first_skill_correct']}/{total}.

Hard gate는 direct semantic JSON, exact sequence, canonicalized strict plan이 각각 12/12일 때만 PASS다. 하나라도 실패하면 production integration에 NOT READY다.

## 계약 경계

모델은 `name`과 `args`만 결정했다. Local registry는 모델이 낸 순서, skill 이름, args를 바꾸지 않고 검증한 뒤에만 `instruction`, `contract`, `budget`을 결정론적으로 채웠다. 알 수 없는 skill, 누락/추가/중복 step, argument key/value 불일치는 즉시 canonicalization 실패이며 정답 sequence로 보정하지 않는다. Parser repair, constrained decoding, fallback은 사용하지 않았고 production runtime도 변경하지 않았다.

## 실행

모델 `{result['model']}` revision `{result['revision']}`을 공식 processor/chat template과 greedy decoding으로 사용했다. 세 보존 reset contact sheet × 의미가 같은 네 instruction, 총 {total} case다. latency min/mean/max는 {summary['latency_ms']['min']:.1f}/{summary['latency_ms']['mean']:.1f}/{summary['latency_ms']['max']:.1f} ms, 관측 peak allocated VRAM은 {summary['peak_vram_mib']:.1f} MiB다. Raw model text는 `raw/`, canonicalized output은 `canonical/`, token/latency/validation 세부 정보는 `cases.json`에 분리 보존했다.
"""
    (out / "REPORT.ko.md").write_text(report, encoding="utf-8")
    provenance = {
        "gpu_host": "amp_csi",
        "gpu": "NVIDIA GeForce RTX 3090 24576 MiB",
        "container_image": "xiaomi-cu121:t_9f03a613",
        "model": result["model"],
        "revision": result["revision"],
        "source_commit": result["source_commit"],
        "input_image_hashes": sorted({case["image_sha256"] for case in cases}),
        "prompt_hashes": sorted({case["prompt_sha256"] for case in cases}),
        "cleanup": None,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
