#!/usr/bin/env python3
"""Raw-generation controls for Xiaomi MiBoT and official Qwen3-VL-Instruct."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_parent_probe(path: Path):
    spec = importlib.util.spec_from_file_location("parent_planning_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eos_ids(model: Any, tokenizer: Any) -> list[int]:
    value = getattr(model.generation_config, "eos_token_id", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return []
    return [int(x) for x in value] if isinstance(value, (list, tuple)) else [int(value)]


def generation_record(*, model: Any, tokenizer: Any, inputs: dict[str, torch.Tensor],
                      max_new_tokens: int, generate_target: Any | None = None) -> dict[str, Any]:
    target = generate_target if generate_target is not None else model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    moved: dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            continue
        moved[key] = value.to(device=device, dtype=dtype if value.is_floating_point() else None)
    input_len = int(moved["input_ids"].shape[-1])
    started = time.monotonic()
    with torch.inference_mode():
        output = target.generate(
            **moved,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    latency_ms = (time.monotonic() - started) * 1000.0
    generated = output.sequences[0, input_len:].detach().cpu().tolist()
    first_topk: list[dict[str, Any]] = []
    if output.scores:
        scores = output.scores[0][0].float().cpu()
        values, indices = torch.topk(scores, k=min(10, scores.numel()))
        first_topk = [
            {"token_id": int(token_id), "logit": float(logit),
             "decoded": tokenizer.decode([int(token_id)], skip_special_tokens=False)}
            for logit, token_id in zip(values, indices)
        ]
    ends = eos_ids(model, tokenizer)
    if generated and generated[-1] in ends:
        stop_reason = "eos_token"
    elif len(generated) >= max_new_tokens:
        stop_reason = "max_new_tokens"
    else:
        stop_reason = "generation_ended_other"
    return {
        "raw_output": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "raw_output_with_special_tokens": tokenizer.decode(generated, skip_special_tokens=False),
        "generated_token_ids": generated,
        "generated_token_count": len(generated),
        "first_generated_token_id": generated[0] if generated else None,
        "first_token_topk": first_topk,
        "eos_token_ids": ends,
        "stop_reason": stop_reason,
        "latency_ms": latency_ms,
        "input_token_count": input_len,
        "input_suffix_token_ids": moved["input_ids"][0, -64:].detach().cpu().tolist(),
        "input_suffix_text": tokenizer.decode(
            moved["input_ids"][0, -64:].detach().cpu().tolist(), skip_special_tokens=False),
    }


def xiaomi_inputs(processor: Any, image: Image.Image, prompt: str,
                  robot_type: str) -> tuple[dict[str, torch.Tensor], str]:
    chat_prompt = (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    state = torch.zeros((1, 4, 60), dtype=torch.float32)
    features = processor(
        images=[image], text=[chat_prompt], state=state,
        robot_type=robot_type, return_tensors="pt",
    )
    keep = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    return {key: features[key] for key in keep if key in features}, chat_prompt


def run_xiaomi(args: argparse.Namespace, parent: Any) -> dict[str, Any]:
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    ).cuda().to(torch.bfloat16).eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    image_path = args.images / "seed0_contact_sheet.png"
    image = Image.open(image_path).convert("RGB")
    prompts = [
        ("qa_object", "What appliance is visible in these camera views? Answer briefly."),
        ("qa_lid_state", "Is the blender lid already securely placed on top of the blender? Answer yes or no."),
        ("qa_scene", "Briefly describe the robot manipulation scene in one sentence."),
        ("planning", parent.full_plan_prompt(parent.INSTRUCTIONS[0])),
    ]
    cases = []
    for case_id, prompt in prompts:
        inputs, chat_prompt = xiaomi_inputs(processor, image, prompt, args.robot_type)
        record = generation_record(
            model=model, tokenizer=tokenizer, inputs=inputs,
            max_new_tokens=args.max_new_tokens, generate_target=model.vlm,
        )
        record.update({
            "case_id": case_id,
            "control": "xiaomi_checkpoint_sanity",
            "prompt": prompt,
            "prompt_sha256": sha256_bytes(prompt.encode()),
            "chat_prompt_sha256": sha256_bytes(chat_prompt.encode()),
            "image_path": str(image_path),
            "image_sha256": sha256_path(image_path),
            "direct_nonempty": bool(record["raw_output"]),
            "direct_valid_json": parent.direct_json(record["raw_output"])[0] is not None,
        })
        cases.append(record)
    return {
        "control": "xiaomi_checkpoint_sanity",
        "model": "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365",
        "revision": args.revision,
        "model_path": args.model,
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "preprocessing": "MiBotProcessor callable with canonical minimal Qwen3-VL image chat prompt, zero state (1,4,60), robot_type=robocasa365",
        "chat_template": "explicit <|im_start|>user + vision tokens + text + <|im_end|> + <|im_start|>assistant",
        "special_tokens": {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "cot_token_ids": tokenizer.encode("<cot></cot>", add_special_tokens=False),
            "cot_open_token_ids": tokenizer.encode("<cot>", add_special_tokens=False),
            "cot_close_token_ids": tokenizer.encode("</cot>", add_special_tokens=False),
            "assistant_prefix_token_ids": tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False),
        },
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "source_commit": args.source_commit,
        "cases": cases,
    }


def official_inputs(processor: Any, image: Image.Image, prompt: str) -> tuple[dict[str, torch.Tensor], str]:
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


def run_official(args: argparse.Namespace, parent: Any) -> dict[str, Any]:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda().eval()
    tokenizer = processor.tokenizer
    prompt = parent.full_plan_prompt(parent.INSTRUCTIONS[0])
    cases = []
    for seed in (0, 1, 2):
        image_path = args.images / f"seed{seed}_contact_sheet.png"
        image = Image.open(image_path).convert("RGB")
        inputs, rendered = official_inputs(processor, image, prompt)
        record = generation_record(
            model=model, tokenizer=tokenizer, inputs=inputs,
            max_new_tokens=args.max_new_tokens,
        )
        parsed, parse_error = parent.direct_json(record["raw_output"])
        score = parent.score_plan(parsed)
        record.update({
            "case_id": f"seed{seed}_full_plan",
            "seed": seed,
            "control": "official_instruct",
            "prompt": prompt,
            "prompt_sha256": sha256_bytes(prompt.encode()),
            "rendered_chat_sha256": sha256_bytes(rendered.encode()),
            "image_path": str(image_path),
            "image_sha256": sha256_path(image_path),
            "direct_nonempty": bool(record["raw_output"]),
            "direct_valid_json": parsed is not None,
            "parse_error": parse_error,
            "direct_valid_plan": score["schema_valid"],
            "sequence": score["sequence"],
            "exact_sequence": score["exact_sequence"],
        })
        cases.append(record)
    return {
        "control": "official_instruct",
        "model": args.model,
        "revision": args.revision,
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "preprocessing": "official AutoProcessor.apply_chat_template with one image and text, add_generation_prompt=True",
        "chat_template": processor.chat_template,
        "special_tokens": {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "assistant_prefix_token_ids": tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False),
        },
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "source_commit": args.source_commit,
        "cases": cases,
    }


def write_outputs(result: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{result['control']}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = []
    for case in result["cases"]:
        rows.append({
            key: json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (list, dict)) else value
            for key, value in case.items() if key not in {"prompt", "first_token_topk"}
        })
    with (out / f"{result['control']}.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sorted({key for row in rows for key in row}), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", choices=("xiaomi", "official"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--parent-probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", default=os.environ.get("SOURCE_COMMIT", "working-tree"))
    parser.add_argument("--robot-type", default="robocasa365")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    parent = load_parent_probe(args.parent_probe)
    result = run_xiaomi(args, parent) if args.control == "xiaomi" else run_official(args, parent)
    write_outputs(result, args.out)


if __name__ == "__main__":
    main()
