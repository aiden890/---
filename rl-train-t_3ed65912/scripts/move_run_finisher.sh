#!/usr/bin/env bash
# MOVE-first run finisher: waits for the detached move_run to produce run_summary.json,
# then writes an honest verdict report + a DONE marker so the run self-completes even if
# the kanban session ends. Mirrors the GRASP-sweep finisher pattern the operator wanted.
set -uo pipefail
cd /home/v4/rl-train-t_3ed65912
run="${1:-move_run1}"
out="results/$run"
summary="$out/run_summary.json"
report="results/MOVE_RUN_REPORT.md"
# wait up to 3h for the run to finish
for _ in $(seq 1 1080); do
  [ -f "$summary" ] && grep -q '"eval_after"' "$summary" 2>/dev/null && break
  # bail if the run process and its container are both gone without a summary
  if ! pgrep -f "grpo_train_loop.py --out /out" >/dev/null 2>&1 \
     && ! docker ps --format '{{.Names}}' | grep -q "xiaomi-client-grpo-$run"; then
    [ -f "$summary" ] || { echo "run died with no summary" > "$out/finisher_note.txt"; break; }
  fi
  sleep 10
done
python3 - "$run" "$summary" "$report" <<'PY'
import json, sys, os
run, summary, report = sys.argv[1], sys.argv[2], sys.argv[3]
if not os.path.exists(summary):
    open(report, "w").write(f"# MOVE run {run}: FAILED (no run_summary.json)\n")
    open(f"results/{run}/DONE", "w").write("nosummary\n"); sys.exit(0)
d = json.load(open(summary))
ph = d.get("phases", {})
b, a = ph.get("eval_before", {}), ph.get("eval_after", {})
pre = d.get("entry_prefilter", {})
br = b.get("skill_success_rate"); ar = a.get("skill_success_rate")
delta = (ar - br) if (br is not None and ar is not None) else None
if delta is None: verdict = "INCONCLUSIVE (missing eval)"
elif delta > 0.05: verdict = f"IMPROVED (+{delta:.3f})"
elif delta < -0.05: verdict = f"REGRESSED ({delta:.3f})"
else: verdict = f"flat ({delta:+.3f})"
curve = ph.get("train_curve", [])
nsucc = [c for c in curve if not c.get("skipped")]
lines = []
lines.append(f"# MOVE-first GRPO run `{run}` — verdict\n")
lines.append(f"**MOVE_HOLDING from grasp-success entry-state, boundary-hold reward.**\n")
lines.append(f"- entry seeds: eval={len(pre.get('eval_seeds',[]))} train={len(pre.get('train_seeds',[]))}\n")
lines.append(f"- EVAL(before): skill_success={br} (n_entered={b.get('n_entered')}/{b.get('n')})")
lines.append(f"- EVAL(after):  skill_success={ar} (n_entered={a.get('n_entered')}/{a.get('n')})")
lines.append(f"- **verdict: {verdict}**\n")
lines.append(f"- train iters run (non-skipped): {len(nsucc)}/{len(curve)}")
if nsucc:
    import statistics as st
    mr = [c["mean_return"] for c in nsucc]
    lines.append(f"- mean_return over trained iters: {min(mr):.3f} .. {max(mr):.3f} (last {nsucc[-1]['mean_return']:.3f})")
lines.append(f"\n## Honest read")
if delta is not None and delta > 0.05:
    lines.append("MOVE-from-entry produced signal AND net improvement -> extend recipe / sweep, "
                 "then consider GRASP/PLACE. Boundary reward triggered (unlike GRASP).")
elif delta is not None:
    lines.append("Boundary reward triggered during training (MOVE succeeds under eta>0, unlike GRASP) "
                 "but net eval improvement was not achieved. Report flat/regressed honestly; "
                 "do NOT dress as success. Next: operator decision (longer/more iters, reward "
                 "weighting, or accept MOVE-from-entry gives signal-but-no-gain at this budget).")
open(report, "w").write("\n".join(lines) + "\n")
open(f"results/{run}/DONE", "w").write(verdict + "\n")
print("WROTE", report, "verdict:", verdict)
PY
echo "finisher done for $run"
