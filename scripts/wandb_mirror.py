#!/usr/bin/env python3
"""wandb 로그 미러러 (방식 B: 서버 무침습).

학습 서버(v4/amp_csi)의 train.log를 SSH/docker로 가져와 파싱하고,
self-hosted wandb(localhost:8080)에 각 arm을 별도 run으로 스트리밍한다.
학습 코드/서버에 손대지 않음 — 이미 찍히는 텍스트 로그만 미러링(준실시간).

멱등: 각 arm의 이미 올린 마지막 step을 state 파일에 기록, 새 iter만 append.
cron에서 주기 실행(예 every 5m). 코드 수정 없이 로그만 읽으므로 학습 간섭 0.
"""
from __future__ import annotations
import json, os, re, subprocess, sys
from pathlib import Path

WANDB_BASE = "http://localhost:8080"
PROJECT = "robocasa-grpo"
STATE_DIR = Path.home() / ".hermes" / "wandb_mirror_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
REPO = Path("/home/aiden/Desktop/lab/robot/robocasa-docker")

# arm 정의: (wandb run 이름, fetch 명령). fetch는 train.log 전체를 stdout으로.
V4_BASE = "/home/v4/rl-train-t_3ed65912/results"
ARMS = {
    # v4 arms: remote.sh로 cat
    "adaptive_adamw3e5_clip02": ("v4", f"{V4_BASE}/adaptive_adamw3e5_clip02/train.log"),
    "adaptive_adamw3e5":        ("v4", f"{V4_BASE}/adaptive_adamw3e5/train.log"),
    "adaptive_sgd2e3":          ("v4", f"{V4_BASE}/adaptive_sgd2e3/train.log"),
    # amp_csi arm: docker exec cat (컨테이너 /train/results)
    "adaptive_adamw3e5_clip03": ("amp", "/train/results/adaptive_adamw3e5_clip03/train.log"),
    "adaptive_adamw5e5":        ("amp", "/train/results/adaptive_adamw5e5/train.log"),
}

def fetch_log(host: str, path: str) -> str | None:
    try:
        if host == "v4":
            cmd = ["bash", str(REPO / "scripts/remote.sh"), f"cat {path} 2>/dev/null"]
        else:  # amp_csi container
            inner = f"docker exec xiaomi-grpo-trainer-t_3ed65912 sh -lc 'cat {path} 2>/dev/null'"
            cmd = ["ssh", "amp_csi", inner]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
        return r.stdout if r.stdout.strip() else None
    except Exception as e:
        print(f"  fetch fail {host}:{path}: {e}", file=sys.stderr)
        return None

# it=27 seed=1175 src=exploit comp=mixed(group_relative) n_succ=6/8 loss=0.0353 ...
TRAIN_RE = re.compile(r"it=(\d+)\s+(.*)")
KV_RE = re.compile(r"(\w+)=([^\s]+)")

def parse_train_line(line: str):
    m = TRAIN_RE.search(line)
    if not m:
        return None
    it = int(m.group(1))
    rest = m.group(2)
    metrics = {"iter": it}
    # n_succ=6/8 -> n_succ, group_size
    ns = re.search(r"n_succ=(\d+)/(\d+)", rest)
    if ns:
        metrics["n_succ"] = int(ns.group(1)); metrics["group_size"] = int(ns.group(2))
        metrics["succ_rate"] = int(ns.group(1)) / int(ns.group(2))
    # GATED?
    metrics["gated"] = 1 if "GATED" in line else 0
    comp = re.search(r"comp=(\w+)", rest)
    if comp: metrics["composition"] = comp.group(1)
    # 나머지 수치 kv (float만)
    for k, v in KV_RE.findall(rest):
        if k in ("seed", "src", "comp", "n_succ", "mem"):
            continue
        try:
            metrics[k] = float(v)
        except ValueError:
            pass
    return it, metrics

def parse_eval(text: str):
    """EVAL(before)/(after): official=x grasp=y 최종 집계 라인."""
    out = {}
    for tag, key in [("EVAL(before)", "before"), ("EVAL(after)", "after")]:
        m = re.search(re.escape(tag) + r":.*?grasp=([\d.]+).*?official=([\d.]+)", text)
        if not m:
            m = re.search(re.escape(tag) + r":.*?official=([\d.]+).*?grasp=([\d.]+)", text)
            if m:
                out[f"eval_{key}_official"] = float(m.group(1)); out[f"eval_{key}_grasp"] = float(m.group(2))
            continue
        out[f"eval_{key}_grasp"] = float(m.group(1)); out[f"eval_{key}_official"] = float(m.group(2))
    return out

def mirror_arm(name: str, host: str, path: str, wandb):
    text = fetch_log(host, path)
    if not text:
        return f"{name}: 로그 없음/미시작"
    # 파싱
    rows = []
    for line in text.splitlines():
        if line.startswith("[train] it=") or re.match(r"^\s*it=\d+", line):
            p = parse_train_line(line)
            if p:
                rows.append(p)
    if not rows:
        return f"{name}: train iter 없음(EVAL 단계)"
    rows.sort(key=lambda x: x[0])
    # state: 마지막 올린 iter
    state_f = STATE_DIR / f"{name}.json"
    last = json.loads(state_f.read_text()).get("last_iter", -1) if state_f.exists() else -1
    new = [(it, m) for it, m in rows if it > last]
    if not new:
        return f"{name}: 신규 없음 (last it={last})"
    # wandb run (arm별 고정 id로 resume)
    run = wandb.init(project=PROJECT, name=name, id=f"grpo-{name}",
                     resume="allow", reinit=True,
                     settings=wandb.Settings(base_url=WANDB_BASE, silent=True))
    ev = parse_eval(text)
    for it, m in new:
        if ev:  # eval 요약은 매 step summary로도 보이게
            m = {**m, **ev}
        run.log(m, step=it)
    if ev:
        run.summary.update(ev)
    run.finish()
    state_f.write_text(json.dumps({"last_iter": new[-1][0]}))
    return f"{name}: +{len(new)} iters (it {new[0][0]}..{new[-1][0]})" + (f" | eval={ev}" if ev else "")

def main():
    os.environ["WANDB_BASE_URL"] = WANDB_BASE
    os.environ["WANDB_SILENT"] = "true"
    import wandb
    results = []
    for name, (host, path) in ARMS.items():
        try:
            results.append(mirror_arm(name, host, path, wandb))
        except Exception as e:
            results.append(f"{name}: ERROR {e}")
    print("\n".join(results))

if __name__ == "__main__":
    main()
