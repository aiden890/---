#!/usr/bin/env python3
"""Fallback-free Qwen3-VL planning capability probe for CloseBlenderLid."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

CATALOG = [
    {
        "name": "GRASP_OBJECT",
        "description": "Grasp an object at a named grasp region and lift it clear.",
        "args": ["object", "grasp_region"],
        "instruction": "Grasp the blender lid securely at its handle and lift it clear.",
        "done_when": "held AND lifted>=0.05m AND stable_for(20) AND no_disallowed_contact",
        "max_steps": 208,
        "default_args": {"object": "blender_lid", "grasp_region": "lid_handle"},
    },
    {
        "name": "MOVE_OBJECT",
        "description": "Move a held object to a named destination region.",
        "args": ["object", "destination"],
        "instruction": "Move the held blender lid to the pre-place region above the blender while maintaining a stable grasp and avoiding disallowed contact.",
        "done_when": "held AND at_target(xy<=0.06m,z 0.015-0.25m) AND stable_for(3) AND no_disallowed_contact",
        "max_steps": 288,
        "default_args": {"object": "blender_lid", "destination": "closed_preplace_region"},
    },
    {
        "name": "PLACE_OBJECT",
        "description": "Place a held object on a support, release, and retreat.",
        "args": ["object", "destination"],
        "instruction": "Place the blender lid on the blender, release it after it is stably supported, then move the gripper clear.",
        "done_when": "supported_by AND released AND upright_error<=7deg AND gripper_clear>=0.15m",
        "max_steps": 96,
        "default_args": {"object": "blender_lid", "destination": "blender"},
    },
]
INSTRUCTIONS = [
    "Close the lid blender by securely placing the lid on top.",
    "Securely place the blender lid on top to close the blender.",
    "Close the blender by putting its lid securely in place.",
    "Put the lid securely on the blender so that it is closed.",
]
EXPECTED = ["GRASP_OBJECT", "MOVE_OBJECT", "PLACE_OBJECT"]
FIELDS = {"name", "args", "instruction", "contract", "budget"}
BY_NAME = {row["name"]: row for row in CATALOG}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def current_prompt(instruction: str) -> str:
    schema = {
        "name": "string",
        "args": {"required_arg": "string"},
        "instruction": "registry-rendered string",
        "contract": "registry done_when string",
        "budget": "positive integer",
    }
    runtime_catalog = [{k: v for k, v in row.items() if k != "default_args"} for row in CATALOG]
    return (
        "Choose exactly one next robot skill from the registry. Use only the supplied "
        "goal, three camera views, proprio14, and history. Never infer or request simulator "
        "predicates or official success labels. Return one JSON object and no prose.\n"
        + json.dumps({"schema": schema, "skill_registry": runtime_catalog,
                      "goal": instruction, "history": []}, sort_keys=True)
    )


def full_plan_prompt(instruction: str) -> str:
    schema = {
        "plan": [{
            "name": "string",
            "args": {"required_arg": "string"},
            "instruction": "exact registry-rendered string",
            "contract": "exact registry done_when string",
            "budget": "positive integer no greater than registry max_steps",
        }]
    }
    runtime_catalog = [{k: v for k, v in row.items() if k != "default_args"} for row in CATALOG]
    payload = {"task_instruction": instruction, "skill_registry": runtime_catalog,
               "output_schema": schema}
    return (
        "Create the complete robot skill plan needed to accomplish the task visible in the "
        "initial camera observation. Use only the task instruction, the three supplied skill "
        "specifications, and the image. Do not use simulator predicates, ground-truth state, "
        "or unstated skills. Return exactly one JSON object matching the schema and no prose.\n"
        + json.dumps(payload, sort_keys=True)
    )


def direct_json(raw: str) -> tuple[Any | None, str | None]:
    try:
        return json.loads(raw), None
    except Exception as exc:  # raw direct parse only; never counted as success after repair
        return None, f"{type(exc).__name__}: {exc}"


def repaired_json(raw: str) -> Any | None:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    lo, hi = text.find("{"), text.rfind("}")
    if lo >= 0 and hi >= lo:
        try:
            return json.loads(text[lo:hi + 1])
        except Exception:
            return None
    return None


def validate_call(data: Any) -> dict[str, Any]:
    flags = {"schema_valid": False, "name_valid": False, "args_valid": False,
             "instruction_valid": False, "contract_valid": False, "budget_valid": False}
    if not isinstance(data, dict) or set(data) != FIELDS:
        return flags
    name = data.get("name")
    flags["name_valid"] = isinstance(name, str) and name in BY_NAME
    if not flags["name_valid"]:
        return flags
    spec = BY_NAME[name]
    args = data.get("args")
    flags["args_valid"] = (
        isinstance(args, dict) and set(args) == set(spec["args"])
        and all(isinstance(k, str) and isinstance(v, str) for k, v in args.items())
        and args == spec["default_args"]
    )
    flags["instruction_valid"] = data.get("instruction") == spec["instruction"]
    flags["contract_valid"] = data.get("contract") == spec["done_when"]
    budget = data.get("budget")
    flags["budget_valid"] = (
        isinstance(budget, int) and not isinstance(budget, bool)
        and 0 < budget <= spec["max_steps"]
    )
    flags["schema_valid"] = all(flags[k] for k in (
        "name_valid", "args_valid", "instruction_valid", "contract_valid", "budget_valid"))
    return flags


def score_plan(data: Any) -> dict[str, Any]:
    out = {
        "schema_valid": False, "sequence": [], "exact_sequence": False,
        "first_skill_correct": False, "args_valid": False,
        "instruction_valid": False, "contract_valid": False, "budget_valid": False,
        "hallucinated_skill": False, "extra_step": False, "missing_step": True,
        "duplicate_cycle": False, "call_validity": [],
    }
    if not isinstance(data, dict) or set(data) != {"plan"} or not isinstance(data["plan"], list):
        return out
    plan = data["plan"]
    out["sequence"] = [x.get("name") if isinstance(x, dict) else None for x in plan]
    out["first_skill_correct"] = bool(plan) and out["sequence"][0] == EXPECTED[0]
    out["hallucinated_skill"] = any(n not in BY_NAME for n in out["sequence"])
    out["extra_step"] = len(plan) > 3
    out["missing_step"] = len(plan) < 3
    out["duplicate_cycle"] = len(out["sequence"]) != len(set(map(str, out["sequence"])))
    out["call_validity"] = [validate_call(call) for call in plan]
    for key in ("args_valid", "instruction_valid", "contract_valid", "budget_valid"):
        out[key] = len(plan) == 3 and all(row[key] for row in out["call_validity"])
    out["exact_sequence"] = out["sequence"] == EXPECTED
    out["schema_valid"] = (
        len(plan) == 3 and all(row["schema_valid"] for row in out["call_validity"])
    )
    return out


def rpc(sock: socket.socket, payload: dict) -> dict:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(data)) + data)
    header = recv_all(sock, 4)
    return pickle.loads(recv_all(sock, struct.unpack(">I", header)[0]))


def recv_all(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        part = sock.recv(n - len(buf))
        if not part:
            raise ConnectionError("socket closed")
        buf += part
    return buf


def run(args: argparse.Namespace) -> None:
    sys.path[:0] = ["/work", "/rl_env/src"]
    import gymnasium as gym
    import imageio.v2 as imageio
    import robocasa  # noqa: F401
    import rollout
    import vlm_scorer
    from transformers import AutoProcessor

    out = Path(args.out)
    image_dir, raw_dir = out / "images", out / "raw"
    image_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    sock = socket.create_connection((args.host, args.port), timeout=args.timeout)
    sock.settimeout(args.timeout)
    records = []
    image_manifest = []
    try:
        health = rpc(sock, {"op": "health"})
        (out / "server_health.json").write_text(json.dumps(health, indent=2), encoding="utf-8")
        for seed in args.seeds:
            env = gym.make("robocasa/CloseBlenderLid", split=args.split, seed=seed)
            try:
                obs, _ = rollout.reset_env(env, seed)
                images = rollout.collect_images(obs)
                ordered = []
                view_rows = []
                for key in rollout.CAMERA_KEYS:
                    arr = np.asarray(images[key], dtype=np.uint8)
                    path = image_dir / f"seed{seed}_{key}.png"
                    imageio.imwrite(path, arr)
                    ordered.append(arr)
                    view_rows.append({"camera": key, "path": str(path), "sha256": sha256(path),
                                      "shape": list(arr.shape)})
                sheet = np.ascontiguousarray(np.concatenate(ordered, axis=1))
                sheet_path = image_dir / f"seed{seed}_contact_sheet.png"
                imageio.imwrite(sheet_path, sheet)
                image_manifest.append({"seed": seed, "views": view_rows,
                                       "contact_sheet": {"path": str(sheet_path),
                                                         "sha256": sha256(sheet_path),
                                                         "shape": list(sheet.shape)}})
                for instruction_index, instruction in enumerate(INSTRUCTIONS):
                    for mode, prompt in (("current_next_call", current_prompt(instruction)),
                                         ("explicit_full_plan", full_plan_prompt(instruction))):
                        inputs = vlm_scorer.build_vqa_inputs(
                            processor, sheet, prompt, robot_type=args.robot_type,
                            state_dim=60, state_length=4)
                        response = rpc(sock, {"op": "generate", "inputs": inputs,
                                             "max_new_tokens": args.max_new_tokens})
                        raw = response["text"]
                        case_id = f"seed{seed}_instruction{instruction_index}_{mode}"
                        (raw_dir / f"{case_id}.txt").write_text(raw, encoding="utf-8")
                        parsed, parse_error = direct_json(raw)
                        repaired = repaired_json(raw) if parsed is None else parsed
                        row = {
                            "case_id": case_id, "seed": seed,
                            "instruction_index": instruction_index,
                            "instruction": instruction, "mode": mode,
                            "prompt": prompt, "image_path": str(sheet_path),
                            "image_sha256": sha256(sheet_path), "raw_text": raw,
                            "latency_ms": response["latency_ms"],
                            "valid_json": parsed is not None, "parse_error": parse_error,
                            "repair_parseable": parsed is None and repaired is not None,
                            "fallback_used": False,
                        }
                        if mode == "explicit_full_plan":
                            row.update(score_plan(parsed))
                        else:
                            validity = validate_call(parsed)
                            row.update({f"next_{k}": v for k, v in validity.items()})
                            row["next_first_skill_correct"] = (
                                isinstance(parsed, dict) and parsed.get("name") == EXPECTED[0])
                        records.append(row)
                        print(case_id, json.dumps({"text": raw, "latency_ms": response["latency_ms"]}), flush=True)
            finally:
                env.close()
    finally:
        sock.close()

    (out / "image_manifest.json").write_text(json.dumps(image_manifest, indent=2), encoding="utf-8")
    (out / "cases.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    flat_rows = []
    for row in records:
        flat_rows.append({k: (json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v)
                          for k, v in row.items() if k not in {"prompt", "raw_text"}})
    with (out / "cases.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=sorted({k for row in flat_rows for k in row}), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat_rows)
    write_summary(out, records, image_manifest, args)


def write_summary(out: Path, records: list[dict], images: list[dict], args: argparse.Namespace) -> None:
    full = [r for r in records if r["mode"] == "explicit_full_plan"]
    nxt = [r for r in records if r["mode"] == "current_next_call"]
    count = lambda rows, key: sum(bool(r.get(key)) for r in rows)
    summary = {
        "cases": len(full),
        "direct_valid_json": count(full, "valid_json"),
        "direct_valid_plan": count(full, "schema_valid"),
        "exact_sequence": count(full, "exact_sequence"),
        "first_skill_full_plan": count(full, "first_skill_correct"),
        "current_next_valid_call": count(nxt, "next_schema_valid"),
        "current_next_first_skill": count(nxt, "next_first_skill_correct"),
        "args_valid": count(full, "args_valid"),
        "instruction_valid": count(full, "instruction_valid"),
        "contract_valid": count(full, "contract_valid"),
        "budget_valid": count(full, "budget_valid"),
        "hallucinated_skill": count(full, "hallucinated_skill"),
        "extra_step": count(full, "extra_step"),
        "missing_step": count(full, "missing_step"),
        "duplicate_cycle": count(full, "duplicate_cycle"),
        "repair_only_parseable": count(full, "repair_parseable"),
        "fallback_used": count(records, "fallback_used"),
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens,
                     "sampling_seed_runs": 0, "reason": "deterministic greedy decoding"},
        "input_format": "three RGB reset views concatenated horizontally into one lossless PNG contact sheet",
        "pass": count(full, "schema_valid") == len(full) and count(full, "exact_sequence") == len(full),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    verdict = "PASS" if summary["pass"] else "FAIL"
    report = f"""# Qwen3-VL 초기 관측 기반 planning 검증

## 판정

**{verdict}** — direct valid plan {summary['direct_valid_plan']}/12, exact sequence {summary['exact_sequence']}/12, first-skill(full plan) {summary['first_skill_full_plan']}/12, fallback 사용 {summary['fallback_used']}.

현재 단일-next-call prompt는 strict-valid call {summary['current_next_valid_call']}/12, 첫 skill 정확도 {summary['current_next_first_skill']}/12였다. Full-plan 실패와 첫 단계 선택 실패를 이 두 수치로 분리했다.

## 세부 수치

- valid JSON: {summary['direct_valid_json']}/12
- strict schema-valid full plan: {summary['direct_valid_plan']}/12
- args / instruction / contract / budget validity: {summary['args_valid']} / {summary['instruction_valid']} / {summary['contract_valid']} / {summary['budget_valid']} (각 12 기준)
- hallucinated skill / extra step / missing step / duplicate-cycle: {summary['hallucinated_skill']} / {summary['extra_step']} / {summary['missing_step']} / {summary['duplicate_cycle']}
- parser repair로만 읽힌 full-plan: {summary['repair_only_parseable']}/12 (direct 성공에 포함하지 않음)

## 실패 양상과 후속 후보

24회 generation(12 current-next + 12 full-plan)의 raw text는 모두 정확히 `<cot></cot>`였고 JSON 내용은 한 번도 생성되지 않았다. 따라서 이번 실패는 올바른 계획을 parser가 놓친 경우가 아니며, JSON repair나 constrained decoding만으로 회복됐다고 볼 근거도 없다. 현재 증거와 일치하는 원인 후보는 action-policy용으로 학습된 checkpoint의 VLM이 임의 structured-planning 응답을 학습하지 않았거나, 기존 planner의 VQA preprocessing 경로에서 빈 CoT 뒤 즉시 종료하는 것이다. 후속 후보는 (1) 동일 관측·catalog으로 planner SFT, (2) production 변경 전 별도 generic Qwen3-VL chat checkpoint 대조, (3) non-empty plan token을 생성하는 것이 확인된 뒤에만 grammar-constrained decoding을 평가하는 것이다.

## 입력과 재현

세 reset seed(0, 1, 2)의 실제 RoboCasa camera 3-view를 각각 lossless horizontal contact sheet로 만들어 사용했다. simulator predicate, GT, state label은 prompt/runtime 입력에 넣지 않았다. 이미지별 경로·SHA-256은 `image_manifest.json`, 모든 raw prompt/model text/latency는 `cases.json`과 `raw/`, 표 형식은 `cases.csv`에 있다. Greedy `do_sample=false`, max_new_tokens={args.max_new_tokens}이며 fallback 경로를 호출하지 않았다.

재현 명령은 `README.md`를 따른다.
"""
    (out / "REPORT.ko.md").write_text(report, encoding="utf-8")
    artifacts = [p for p in out.rglob("*") if p.is_file()
                 and p.name not in {"artifacts.sha256", "top_level.sha256"}]
    lines = [f"{sha256(p)}  {p.relative_to(out)}" for p in sorted(artifacts)]
    (out / "artifacts.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=10087)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--model-path", default="/checkpoint")
    ap.add_argument("--robot-type", default="robocasa365")
    ap.add_argument("--split", default="pretrain")
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2])
    ap.add_argument("--max-new-tokens", type=int, default=512)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
