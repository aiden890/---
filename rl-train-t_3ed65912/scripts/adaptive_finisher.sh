#!/usr/bin/env bash
# adaptive_finisher.sh — waits for BOTH adaptive-sweep arms (v4 local adaptive_sgd2e3 +
# amp_csi remote adaptive_adamw1e5) to drop their .DONE sentinel, pulls the amp_csi arm's
# artifacts to v4, then aggregates a REPORT (GATED-ratio drop vs baseline, mixed% curve,
# before/after eval delta, curriculum band-drift + explore/exploit split, post-step
# metrics) and writes results/adaptive/ADAPTIVE_FINAL.md + a top-level DONE sentinel.
# Detached-run on v4 alongside adaptive_sweep.sh so the kanban worker can park and collect.
#
# Usage (on v4, detached):
#   setsid nohup bash scripts/adaptive_finisher.sh > results/adaptive/finisher.nohup 2>&1 &
set -uo pipefail
train="/home/v4/rl-train-t_3ed65912"
cd "$train"
base="$train/results/adaptive"; mkdir -p "$base"
# Parametrized so the SAME finisher serves later LR-ladder sweeps (anti-duplication).
V4_RUN="${V4_RUN:-adaptive_sgd2e3}"
AMP_RUN="${AMP_RUN:-adaptive_adamw1e5}"
V4_LABEL="${V4_LABEL:-SGD 2e-3}"
AMP_LABEL="${AMP_LABEL:-AdamW 1e-5}"
FINAL_MD="${FINAL_MD:-$base/ADAPTIVE_FINAL.md}"
flog="$base/finisher.log"; : > "$flog"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$flog"; }

# amp_csi (guest) ssh: same key/host as band_finisher.sh. The sweep launcher on amp_csi
# writes results/adaptive/adaptive_adamw1e5.DONE on that box.
AMP="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -i /home/v4/.ssh/csi_gyuri -p 10001 guest@115.145.173.248"
AMP_BASE="/home/guest/rl-train-t_3ed65912/results/adaptive"

V4_DONE="$base/${V4_RUN}.DONE"
log "waiting for v4 arm ($V4_DONE) + amp_csi arm (remote ${AMP_RUN}.DONE) ..."
for i in $(seq 1 720); do   # up to 12h (60s * 720)
  v4ok=0; ampok=0
  [ -f "$V4_DONE" ] && v4ok=1
  $AMP "test -f $AMP_BASE/${AMP_RUN}.DONE" 2>/dev/null && ampok=1
  if [ "$v4ok" = 1 ] && [ "$ampok" = 1 ]; then log "both arms DONE"; break; fi
  sleep 60
done

# pull amp_csi arm artifacts to v4 for a single aggregation point
mkdir -p "$base/$AMP_RUN"
for f in run_summary.json train_log.jsonl eval_before.json eval_after.json eval_heldout.json seed_difficulty.json; do
  $AMP "cat $AMP_BASE/$AMP_RUN/$f" > "$base/$AMP_RUN/$f" 2>/dev/null || true
done

log "aggregating REPORT ..."
python3 "$train/scripts/adaptive_aggregate.py" "$base" \
  "${V4_RUN}:${V4_LABEL}:v4" "${AMP_RUN}:${AMP_LABEL}:amp_csi" > "$FINAL_MD" 2>>"$flog" || \
  log "aggregate script failed (see $flog)"
echo "OK $(date +%s)" > "$base/ADAPTIVE_ALL.DONE"
log "== adaptive finisher DONE -> $FINAL_MD =="
