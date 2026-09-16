#!/usr/bin/env bash
# band_finisher.sh — waits for BOTH band-sweep arms (v4 local band_sgd2e3 + amp_csi remote
# band_adamw1e5) to drop their .DONE sentinel, pulls the amp_csi arm's artifacts to v4, then
# aggregates a REPORT (GATED-ratio drop vs baseline, before/after eval delta, post-step
# metrics) and writes results/band/BAND_FINAL.md + a top-level DONE sentinel. Detached-run
# on v4 alongside band_sweep.sh so the kanban worker can park and collect later.
#
# Usage (on v4, detached):
#   setsid nohup bash scripts/band_finisher.sh > results/band/finisher.nohup 2>&1 &
set -uo pipefail
train="/home/v4/rl-train-t_3ed65912"
cd "$train"
base="$train/results/band"; mkdir -p "$base"
flog="$base/finisher.log"; : > "$flog"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$flog"; }

# amp_csi (guest) ssh: key from run-train.sh's remote path pattern; the sweep launcher on
# amp_csi writes results/band/band_adamw1e5.DONE on that box.
AMP="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -i /home/v4/.ssh/csi_gyuri -p 10001 guest@115.145.173.248"
AMP_BASE="/home/guest/rl-train-t_3ed65912/results/band"

V4_DONE="$base/band_sgd2e3.DONE"
log "waiting for v4 arm ($V4_DONE) + amp_csi arm (remote band_adamw1e5.DONE) ..."
for i in $(seq 1 720); do   # up to 12h (60s * 720)
  v4ok=0; ampok=0
  [ -f "$V4_DONE" ] && v4ok=1
  $AMP "test -f $AMP_BASE/band_adamw1e5.DONE" 2>/dev/null && ampok=1
  if [ "$v4ok" = 1 ] && [ "$ampok" = 1 ]; then log "both arms DONE"; break; fi
  sleep 60
done

# pull amp_csi arm artifacts to v4 for a single aggregation point
mkdir -p "$base/band_adamw1e5"
for f in run_summary.json train_log.jsonl eval_before.json eval_after.json eval_heldout.json train_pool.json; do
  $AMP "cat $AMP_BASE/band_adamw1e5/$f" > "$base/band_adamw1e5/$f" 2>/dev/null || true
done

log "aggregating REPORT ..."
python3 "$train/scripts/band_aggregate.py" "$base" > "$base/BAND_FINAL.md" 2>>"$flog" || \
  log "aggregate script failed (see $flog)"
echo "OK $(date +%s)" > "$base/BAND_ALL.DONE"
log "== band finisher DONE -> $base/BAND_FINAL.md =="
