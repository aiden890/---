#!/usr/bin/env bash
# G7 (post-update ODE transfer) self-completing finisher for the pi-RL sampler.
#
# The `train` subcommand already implements G7: EVAL(eta=0 ODE) -> TRAIN(eta>0 pi-RL
# rollouts + PPO/GRPO update) -> EVAL(eta=0 ODE) on the SAME paired eval pool, then
# HELDOUT. So "post-update ODE transfer" = the before/after eta=0 delta on that pool.
#
# This finisher waits for a detached `train` run to finish (its train.log ends with
# "=== DONE ==="), then writes an HONEST verdict to REPORT/G7_<run>_FINAL.md and a
# DONE sentinel. Runs detached so the session need not stay open.
#
# Usage (on v4): nohup bash scripts/g7_finisher.sh <run_subdir> > results/<run>/finisher.log 2>&1 &
set -uo pipefail
train=/home/v4/rl-train-t_3ed65912
run="${1:?run subdir required}"
out="$train/results/$run"
log="$out/train.log"
report="$train/REPORT/G7_${run}_FINAL.md"

# wait up to 6h for the run to finish
for i in $(seq 1 2160); do
  if [ -f "$log" ] && grep -q "=== DONE ===" "$log" 2>/dev/null; then break; fi
  # bail out early if the client container died without DONE
  if [ "$i" -gt 3 ] && ! docker ps --format '{{.Names}}' | grep -q "xiaomi-client-grpo-$run"; then
    if ! grep -q "=== DONE ===" "$log" 2>/dev/null; then
      echo "container gone before DONE at check $i" >> "$out/finisher.log"; break
    fi
  fi
  sleep 10
done

python3 - "$out" "$report" "$run" <<'PY'
import json, sys, pathlib
out, report, run = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
def load(p):
    f = out / p
    return json.loads(f.read_text()) if f.exists() else None
before, after, heldout = load("eval_before.json"), load("eval_after.json"), load("eval_heldout.json")
summ = load("run_summary.json") or {}
cfg = summ.get("config", {})
def rate(d):
    if not d: return None
    # entry-mode uses skill_success_rate; full-plan uses official/grasp
    for k in ("skill_success_rate", "official_success_rate", "grasp_success_rate"):
        if k in d and d[k] is not None: return d[k], k
    return None
rb, ra, rh = rate(before), rate(after), rate(heldout)
def fmt(r): return f"{r[0]} ({r[1]})" if r else "n/a"
delta = (ra[0]-rb[0]) if (rb and ra) else None
verdict = "n/a"
if delta is not None:
    verdict = "IMPROVED" if delta > 1e-9 else ("flat" if abs(delta) <= 1e-9 else "REGRESSED")
lines = [
 f"# G7 post-update ODE transfer — `{run}`", "",
 "**정의:** eta=0 결정론 ODE(=체크포인트 sampler) eval을 pi-RL(eta>0) rollout+PPO/GRPO 업데이트 전후로 측정.",
 "before==after는 policy delta 0을 의미(adapter는 skill 주어지면 항상 활성 — audit fix).", "",
 f"- sampler: pirl  | skill: {cfg.get('train_skill')}  | eta(train): {cfg.get('eta')}",
 f"- iters: {cfg.get('iters')}  group: {cfg.get('group')}  update_epochs: {cfg.get('update_epochs')}  eval_n: {cfg.get('eval_n')}",
 f"- entry_from_grasp: {cfg.get('entry_from_grasp')}  reward_variant: {cfg.get('reward_variant')}", "",
 "| phase | success rate |", "|---|---|",
 f"| EVAL(before, eta=0 ODE) | {fmt(rb)} |",
 f"| EVAL(after,  eta=0 ODE) | {fmt(ra)} |",
 f"| HELDOUT (eta=0 ODE)     | {fmt(rh)} |", "",
 f"**verdict: {verdict}" + (f" ({delta:+.4f})" if delta is not None else "") + "**", "",
 "## 정직한 판독",
]
# train-curve signal check
curve = (summ.get("phases", {}) or {}).get("train_curve", [])
nsucc = None
try:
    import statistics
    # look at train_log.jsonl for n_success_hold if present
    jl = out / "train_log.jsonl"
    if jl.exists():
        rows = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()]
        hulls = [r.get("n_success_hold") for r in rows if r.get("n_success_hold") is not None]
        skipped = sum(1 for r in rows if r.get("skipped"))
        nsucc = (sum(x for x in hulls if x), len(rows), skipped)
except Exception:
    pass
if nsucc:
    lines.append(f"- train rollout signal: total n_success_hold={nsucc[0]} over {nsucc[1]} iters "
                 f"(skipped={nsucc[2]}). 0이면 eta>0 rollout서 성공 부재 → GRPO 신호 없음(붕괴/무개선의 원인이 method가 아니라 신호부재).")
if verdict == "flat":
    lines.append("- flat: 업데이트가 eta=0 ODE 행동을 바꾸지 못함. 신호부재(위 n_success_hold) 또는 lr/iters 부족.")
elif verdict == "REGRESSED":
    lines.append("- REGRESSED: 업데이트가 base 대비 성공률을 낮춤. adapter가 base 역량을 덮어씀 — lr/clip/kl 재검토.")
elif verdict == "IMPROVED":
    lines.append("- IMPROVED: pi-RL 경로로 eta=0 ODE 성공률이 실제 향상 — MOVE/RELEASE 확장 대상.")
pathlib.Path(report).write_text("\n".join(lines) + "\n")
(out / "DONE").write_text(f"{verdict} delta={delta}\n")
print("wrote", report, "verdict", verdict)
PY
echo "G7 finisher done: $report" >> "$out/finisher.log"
