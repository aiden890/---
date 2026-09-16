#!/usr/bin/env bash
# Stage-1 finisher (task t_74a6ed7a): waits for BOTH servers' arm sweeps to finish, then
# aggregates all four arms and writes an HONEST verdict + selection to REPORT/STAGE1_FINAL.md
# and a DONE sentinel. Runs on v4 (which rsyncs amp_csi's results in), detached, so the
# session need not stay open. Selection criterion (reference doc): paired GRASP success
# improvement (after - before on the SAME eval pool) AND post-step ratio/KL/clip stability.
#
# Usage (on v4): nohup bash scripts/stage1_finisher.sh > results/stage1/finisher.log 2>&1 &
set -uo pipefail
train=/home/v4/rl-train-t_3ed65912
base="$train/results/stage1"
KEY=/home/v4/.ssh/csi_gyuri
RSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -i $KEY -p 10001"
DST=guest@115.145.173.248
DSTBASE=/home/guest/rl-train-t_3ed65912/results/stage1
report="$train/REPORT/STAGE1_FINAL.md"
mkdir -p "$base"
log(){ echo "[$(date +%H:%M:%S)] $*" >> "$base/finisher.log"; }

# wait up to 12h for v4's DONE and amp_csi's DONE
log "waiting for v4 + amp_csi Stage-1 DONE sentinels..."
for i in $(seq 1 4320); do
  v4done=0; ampdone=0
  [ -f "$base/v4.DONE" ] && v4done=1
  # pull amp_csi's DONE + results into the local stage1 tree
  $RSH $DST "test -f $DSTBASE/ampcsi.DONE" 2>/dev/null && ampdone=1
  if [ "$ampdone" = 1 ]; then
    rsync -a -e "$RSH" "$DST:$DSTBASE/" "$base/" >>"$base/finisher.log" 2>&1 || true
  fi
  [ "$v4done" = 1 ] && [ "$ampdone" = 1 ] && break
  sleep 10
done
# final pull of amp_csi results regardless
$RSH $DST "test -d $DSTBASE" 2>/dev/null && rsync -a -e "$RSH" "$DST:$DSTBASE/" "$base/" >>"$base/finisher.log" 2>&1 || true

log "aggregating arms..."
python3 - "$base" "$report" <<'PY'
import json, sys, pathlib, datetime
base, report = pathlib.Path(sys.argv[1]), sys.argv[2]

def load(p):
    return json.loads(p.read_text()) if p.exists() else None

rows = []
for d in sorted(base.iterdir()):
    if not d.is_dir():
        continue
    summ = load(d / "run_summary.json")
    before = load(d / "eval_before.json")
    after = load(d / "eval_after.json")
    heldout = load(d / "eval_heldout.json")
    if summ is None and before is None:
        continue
    tc = (summ or {}).get("trainer_config", {}) or {}
    cfg = tc.get("config", {}) or {}
    curve = ((summ or {}).get("phases", {}) or {}).get("train_curve", []) or []
    # post-step diagnostics: mean over non-gated iters
    def _avg(key):
        vals = [c.get(key) for c in curve if isinstance(c.get(key), (int, float)) and not c.get("gated")]
        return round(sum(vals) / len(vals), 4) if vals else None
    n_gated = sum(1 for c in curve if c.get("gated"))
    comp = {}
    for c in curve:
        k = c.get("group_composition")
        comp[k] = comp.get(k, 0) + 1
    def grasp(x):
        return (x or {}).get("grasp_success_rate")
    rows.append({
        "arm": d.name, "optimizer": tc.get("optimizer") or cfg.get("optimizer"),
        "lr": cfg.get("lr"), "trainable": tc.get("trainable_params"),
        "grasp_before": grasp(before), "grasp_after": grasp(after), "grasp_heldout": grasp(heldout),
        "official_before": (before or {}).get("official_success_rate"),
        "official_after": (after or {}).get("official_success_rate"),
        "delta_grasp": (None if grasp(after) is None or grasp(before) is None
                        else round(grasp(after) - grasp(before), 4)),
        "post_ratio": _avg("post_step_mean_ratio"), "post_kl": _avg("post_step_mean_kl"),
        "post_clip": _avg("post_step_clip_fraction"), "post_ess": _avg("post_step_ess"),
        "adapter_dL2": _avg("adapter_delta_l2"),
        "n_iters": len(curve), "n_gated": n_gated, "composition": comp,
    })

# selection: largest positive paired GRASP delta among arms with stable post-step
# (ratio in [0.5, 2.0], no non-finite). Report honestly if none improved.
def stable(r):
    pr = r["post_ratio"]
    return pr is not None and 0.5 <= pr <= 2.0
cands = [r for r in rows if r["delta_grasp"] is not None]
improved = [r for r in cands if r["delta_grasp"] > 0 and stable(r)]
best = max(improved, key=lambda r: r["delta_grasp"]) if improved else (
       max(cands, key=lambda r: (r["delta_grasp"])) if cands else None)

lines = [f"# Stage-1 optimizer/LR screening — FINAL ({datetime.datetime.now().isoformat(timespec='seconds')})", ""]
lines.append("Common: pi-RL noise 0.1, group 8, update_epochs 1, 30 iter, clip 0.1, grad_clip 0.5,")
lines.append("paired train/eval seeds, eval 20 paired + 20 held-out, reward=terminal_plus_hold, horizon_grasp 208.")
lines.append("Primary metric: paired GRASP success (after - before, SAME eval pool). Advantage/measurement fixes applied.")
lines.append("")
lines.append("| arm | opt | lr | GRASP before→after (Δ) | heldout | official b→a | post ratio/KL/clip | adapter ΔL2 | gated iters | composition |")
lines.append("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    lines.append(
        f"| {r['arm']} | {r['optimizer']} | {r['lr']} | "
        f"{r['grasp_before']}→{r['grasp_after']} ({r['delta_grasp']}) | {r['grasp_heldout']} | "
        f"{r['official_before']}→{r['official_after']} | "
        f"{r['post_ratio']}/{r['post_kl']}/{r['post_clip']} | {r['adapter_dL2']} | "
        f"{r['n_gated']}/{r['n_iters']} | {r['composition']} |")
lines.append("")
if best is None:
    lines.append("**verdict: NO arm produced a comparable paired result (missing eval).**")
elif not improved:
    lines.append(f"**verdict: NO arm improved paired GRASP success. Best (least-bad) = {best['arm']} "
                 f"(Δ={best['delta_grasp']}). Report honestly; do NOT advance to Stage 2 on a non-improving "
                 f"screen — escalate to operator for a difficulty-adjusted target or shaped-advantage.**")
else:
    lines.append(f"**verdict: SELECT {best['arm']} (optimizer={best['optimizer']}, lr={best['lr']}) — "
                 f"paired GRASP Δ={best['delta_grasp']} with stable post-step ratio={best['post_ratio']}. "
                 f"Proceed to Stage 2 (epochs 1 vs 2, clip 0.1 vs 0.2) with this arm.**")
pathlib.Path(report).write_text("\n".join(lines) + "\n")
print("wrote", report)
(base / "STAGE1_AGG.json").write_text(json.dumps({"rows": rows, "selected": best}, indent=2, default=str))
PY

echo "OK $(date +%s)" > "$base/STAGE1_FINAL.DONE"
log "== Stage-1 finisher DONE -> REPORT/STAGE1_FINAL.md =="
