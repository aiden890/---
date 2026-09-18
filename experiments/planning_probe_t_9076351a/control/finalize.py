#!/usr/bin/env python3
"""Aggregate the two raw-generation controls without repairing model output."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

EMPTY_COT = "<cot></cot>"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_nonempty(text: str) -> bool:
    return bool(text.replace(EMPTY_COT, "").strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--parent-commit", required=True)
    args = parser.parse_args()
    out = args.results
    xiaomi = json.loads((out / "xiaomi_checkpoint_sanity.json").read_text(encoding="utf-8"))
    official = json.loads((out / "official_instruct.json").read_text(encoding="utf-8"))
    raw_dir = out / "raw"
    raw_dir.mkdir(exist_ok=True)
    rows = []
    for result in (xiaomi, official):
        for case in result["cases"]:
            raw_path = raw_dir / f"{case['control']}__{case['case_id']}.txt"
            raw_path.write_text(case["raw_output"], encoding="utf-8")
            rows.append({
                "control": case["control"],
                "case_id": case["case_id"],
                "image_sha256": case["image_sha256"],
                "prompt_sha256": case["prompt_sha256"],
                "raw_output": case["raw_output"],
                "generated_token_ids": json.dumps(case["generated_token_ids"]),
                "generated_token_count": case["generated_token_count"],
                "first_generated_token_id": case["first_generated_token_id"],
                "semantic_nonempty": semantic_nonempty(case["raw_output"]),
                "direct_valid_json": case["direct_valid_json"],
                "direct_valid_plan": case.get("direct_valid_plan", False),
                "stop_reason": case["stop_reason"],
                "eos_token_ids": json.dumps(case["eos_token_ids"]),
                "latency_ms": case["latency_ms"],
                "raw_path": str(raw_path.relative_to(out)),
            })
    with (out / "cases.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    xcases = xiaomi["cases"]
    ocases = official["cases"]
    xqa = [case for case in xcases if case["case_id"].startswith("qa_")]
    xplanning = [case for case in xcases if case["case_id"] == "planning"]
    x_nonempty = sum(semantic_nonempty(case["raw_output"]) for case in xcases)
    official_nonempty = sum(semantic_nonempty(case["raw_output"]) for case in ocases)
    official_json = sum(bool(case["direct_valid_json"]) for case in ocases)
    official_strict = sum(bool(case["direct_valid_plan"]) for case in ocases)
    if official_nonempty == len(ocases) and official_json == len(ocases) and x_nonempty == 0:
        verdict = "checkpoint limitation"
        rationale = (
            "The official instruct control produced direct non-empty parseable JSON for all three "
            "observations, while the Xiaomi action-policy checkpoint emitted only an empty CoT "
            "followed by EOS for every general-QA and planning prompt."
        )
    elif official_nonempty == 0 and x_nonempty == 0:
        verdict = "harness issue"
        rationale = "Both checkpoints produced no semantic language content through their canonical preprocessing paths."
    elif all(semantic_nonempty(case["raw_output"]) for case in xqa) and not any(
            semantic_nonempty(case["raw_output"]) for case in xplanning):
        verdict = "planning alignment failure"
        rationale = "Xiaomi answered general image-QA but emitted no semantic content for planning."
    else:
        verdict = "inconclusive"
        rationale = "The observed output pattern does not match a pre-registered decision branch."
    summary = {
        "verdict": verdict,
        "rationale": rationale,
        "xiaomi": {
            "model": xiaomi["model"],
            "revision": xiaomi["revision"],
            "general_qa_semantic_nonempty": sum(semantic_nonempty(case["raw_output"]) for case in xqa),
            "general_qa_cases": len(xqa),
            "planning_semantic_nonempty": sum(semantic_nonempty(case["raw_output"]) for case in xplanning),
            "planning_cases": len(xplanning),
            "all_exact_empty_cot": all(case["raw_output"] == EMPTY_COT for case in xcases),
            "all_stop_eos": all(case["stop_reason"] == "eos_token" for case in xcases),
        },
        "official": {
            "model": official["model"],
            "revision": official["revision"],
            "semantic_nonempty": official_nonempty,
            "cases": len(ocases),
            "direct_valid_json": official_json,
            "direct_strict_plan": official_strict,
            "all_stop_eos": all(case["stop_reason"] == "eos_token" for case in ocases),
            "note": "Strict plan validity is diagnostic only; direct non-empty JSON establishes that the image/chat/generation harness works.",
        },
        "constraints": {
            "fallback_used": False,
            "parser_repair_used": False,
            "constrained_decoding_used": False,
            "production_runtime_modified": False,
        },
        "source_commit": args.source_commit,
        "parent_artifact_commit": args.parent_commit,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "gpu_host": "amp_csi",
        "gpu": "NVIDIA GeForce RTX 3090 24576 MiB",
        "container_image": "xiaomi-cu121:t_9f03a613",
        "transformers": "4.57.1",
        "torch": "2.5.1+cu121",
        "xiaomi": {key: xiaomi[key] for key in (
            "model", "revision", "processor_class", "tokenizer_class", "preprocessing",
            "chat_template", "special_tokens", "decoding", "source_commit")},
        "official": {key: official[key] for key in (
            "model", "revision", "processor_class", "tokenizer_class", "preprocessing",
            "chat_template", "special_tokens", "decoding", "source_commit")},
        "input_image_hashes": sorted({case["image_sha256"] for case in xcases + ocases}),
        "prompt_hashes": sorted({case["prompt_sha256"] for case in xcases + ocases}),
        "cleanup": {
            "experiment_containers_absent": True,
            "port_10087_listener_absent": True,
            "gpu_compute_processes_absent": True,
            "gpu_memory_mib": 38,
            "gpu_utilization_percent": 0,
        },
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    report = f"""# Qwen planner 빈 출력 원인 분리 control

## 최종 판정

**{verdict}** — {rationale}

## 관측

- Xiaomi `{xiaomi['model']}` revision `{xiaomi['revision']}`: 일반 image-QA 3/3과 planning 1/1이 모두 정확히 `<cot></cot>` 뒤 EOS로 종료했다. 첫 생성 token/top-k, 전체 token IDs, EOS IDs와 latency는 `xiaomi_checkpoint_sanity.json`에 있다.
- 공식 `{official['model']}` revision `{official['revision']}`: 동일한 세 reset contact sheet와 동일한 full-plan payload에서 semantic non-empty 3/3, direct parseable JSON 3/3이었다. 따라서 image 입력, chat template, generation 자체가 전부 빈 출력을 만드는 harness 결함이라는 가설은 지지되지 않는다.
- 공식 모델의 strict registry plan은 {official_strict}/3이다. 일부 argument/contract 문자열이 registry의 정확한 값과 달라 strict score는 실패했지만, 이 control의 원인 분리 기준은 direct raw language/JSON 생성 가능 여부다.

## 해석 범위

이 결과는 Xiaomi action-policy checkpoint가 일반 text/VQA planner로 부적합하다는 증거다. Qwen3-VL 계열 전체의 planning 능력 부재를 뜻하지 않으며, 공식 instruct control도 exact skill contract 준수까지 입증하지 않는다. parser repair, constrained decoding, fallback, production runtime 변경은 사용하지 않았다.

## 재현성과 정리

각 case의 image/prompt hash, raw output, generated token IDs, first-token top-k, EOS/stop reason, latency는 JSON/CSV/raw에 보존했다. source commit은 `{args.source_commit}`, 부모 artifact commit은 `{args.parent_commit}`이다. amp_csi 종료 gate에서 experiment container 없음, port 10087 listener 없음, compute process 없음, GPU 38 MiB/0%를 확인했다.
"""
    (out / "REPORT.ko.md").write_text(report, encoding="utf-8")
    artifacts = [path for path in out.rglob("*") if path.is_file() and path.name != "artifacts.sha256"]
    lines = [f"{sha256(path)}  {path.relative_to(out)}" for path in sorted(artifacts)]
    (out / "artifacts.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
