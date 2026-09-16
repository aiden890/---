#!/usr/bin/env bash
# band_sweep.sh — difficulty-band-prefilter GRASP GRPO sweep (task t_2907de4f).
# Redesign of the pre-filter baseline (s1_sgd2e3 / s1_adamw1e5): now every train seed is
# BASE-difficulty-band prefiltered (--train-difficulty-band LOW HIGH) so all-success(8/8)
# and all-failure(0/8) GATED seeds are dropped, and the train loop draws WITHOUT
# replacement from the band pool (--seed) for diversity. eval/heldout stay FIXED (not
# band-filtered) so before/after delta is a clean policy-improvement measure.
#
# ONE arm per server -> v4 + amp_csi run in TRUE parallel (unlike stage1_sweep which ran
# arms sequentially). Paired: both arms share band/scan-cap/group/iters/eval-seeds/--seed,
# differing ONLY in optimizer+LR, so the comparison is the pre-filter baseline's SGD-2e-3
# vs AdamW-1e-5 optimizer contrast, re-run WITH the band prefilter.
#
# Usage (on the server; RC_ROOT=/home/guest for amp_csi):
#   nohup bash scripts/band_sweep.sh <run> <opt> <lr> \
#         > results/band/<run>.nohup 2>&1 &
#   e.g. v4:      band_sweep.sh band_sgd2e3   sgd   2e-3
#        amp_csi: band_sweep.sh band_adamw1e5 adamw 1e-5
set -uo pipefail
ROOT="${RC_ROOT:-/home/v4}"
train="$ROOT/rl-train-t_3ed65912"
cd "$train"
run="${1:?run name required}"; opt="${2:?optimizer required}"; lr="${3:?lr required}"
base="$train/results/band"; mkdir -p "$base"
out="$base/$run"; mkdir -p "$out"
slog="$base/${run}.log"; : > "$slog"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$slog"; }

# Band prefilter config (operator: band 1 7, scan-cap generous, group 8, iters 30, fixed
# eval seeds). horizon 208 / terminal_plus_hold / eta 0.1 kept from the baseline. The
# --train-* band args + --seed are grpo_train_loop CLIENT args; optimizer/lr/sampler/eta/
# grad-clip are TRAINER-SERVER args (set at trainer-start).
COMMON="--train-skill grasp --eta 0.1 --group 8 --iters 30 --update-epochs 1 \
--clip 0.1 --hold-steps 20 --reward-variant terminal_plus_hold \
--horizon-grasp 208 --eval-n 20 --heldout-n 20 --save-videos 6 \
--train-difficulty-band 1 7 --train-scan-cap 240 --train-pool-need 45 --seed 12345"

log "== BAND sweep arm $run: optimizer=$opt lr=$lr (band 1 7, scan-cap 240, pool 45) =="

# clean any stale client of this name + any prior trainer, then start the trainer with
# THIS arm's optimizer/lr (pi-RL sampler, eta 0.1). Same server-arg set as the baseline.
docker rm -f "xiaomi-client-grpo-$run" >>"$slog" 2>&1 || true
bash scripts/run-train.sh trainer-stop >>"$slog" 2>&1 || true
bash scripts/run-train.sh trainer-start \
  --sampler pirl --eta 0.1 --optimizer "$opt" --lr "$lr" \
  --grad-clip 0.5 --clip 0.1 --update-epochs 1 >>"$slog" 2>&1 \
  || { log "trainer-start failed for $run"; exit 3; }
sleep 20  # model load

log "train $run ... (band prefilter runs INSIDE grpo_train_loop before EVAL(before))"
bash scripts/run-train.sh train "$run" $COMMON >>"$out/arm.log" 2>&1
rc=$?
# run-train.sh writes results/$run ; copy the key artifacts under band/$run for aggregation
if [ -d "$train/results/$run" ] && [ "$train/results/$run" != "$out" ]; then
  for f in run_summary.json train_log.jsonl eval_before.json eval_after.json \
           eval_heldout.json train_pool.json; do
    cp -f "$train/results/$run/$f" "$out/$f" 2>/dev/null || true
  done
fi
log "arm $run finished rc=$rc"
bash scripts/run-train.sh trainer-stop >>"$slog" 2>&1 || true

echo "OK $(date +%s) rc=$rc" > "$base/${run}.DONE"
log "== BAND sweep arm $run DONE (rc=$rc) =="
