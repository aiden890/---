#!/usr/bin/env bash
# wait-then-run.sh (task t_fc5e73d5) -- run the obs-only VLM inference the moment
# amp_csi's GPU is free of the stage1 GRPO sweep, WITHOUT disturbing training.
#
# Detached (setsid nohup) so it survives the Hermes session. Polls until:
#   * no stage1_sweep.sh / run-train.sh process is running for guest, AND
#   * GPU used memory < FREE_MB (training container gone),
# then launches run-vlm-inference.sh and writes a DONE sentinel.
set -uo pipefail

pkg=/home/guest/architecture-smoke-t_fc5e73d5
seeds="${1:-0,1,2,3,4,5}"
free_mb="${2:-3000}"
log="$pkg/wait_then_run.log"
sentinel="$pkg/INFERENCE.DONE"
mkdir -p "$pkg"

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$log"; }

say "watcher start: waiting for amp_csi GPU free (< ${free_mb}MB) and no stage1 sweep"
while true; do
  busy_proc=$(ps -u guest -o args= 2>/dev/null | grep -E "stage1_sweep|run-train.sh" | grep -v grep | wc -l)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$busy_proc" -eq 0 ] && [ "${used:-99999}" -lt "$free_mb" ]; then
    say "GPU free (used=${used}MB, sweep_procs=${busy_proc}) -> launching inference"
    break
  fi
  say "still busy (used=${used}MB, sweep_procs=${busy_proc}); sleep 120"
  sleep 120
done

cd "$pkg/architecture_smoke"
bash run-vlm-inference.sh "$seeds" >>"$log" 2>&1
rc=$?
say "run-vlm-inference.sh exit=$rc"
echo "exit=$rc $(date +%s)" > "$sentinel"
say "watcher done -> sentinel $sentinel written"
