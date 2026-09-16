#!/usr/bin/env bash
# adaptive_sweep.sh — ONLINE adaptive-curriculum GRASP GRPO sweep (task t_2907de4f).
#
# Operator redesign (supersedes band_sweep.sh's upfront prefilter scan): there is NO
# separate difficulty scan. Each train iteration's group rollout ALREADY measures how hard
# its seed is for the CURRENT policy (n_succ of `group` stochastic members); the
# AdaptiveCurriculum sampler folds that free signal into a per-seed difficulty EMA and
# biases the NEXT iter's seed toward the MID band [LOW,HIGH] (mixed group = real GRPO
# advantage; 0/8 & 8/8 are GATED and dropped). The band MOVES with the policy: mastered
# seeds leave, newly-mid seeds enter. Every iter still draws a DIFFERENT seed (overfit
# guard). eval/heldout stay FIXED (not curriculum-filtered) so before/after delta is a
# clean policy-improvement measure.
#
# ONE arm per server -> v4 + amp_csi run in TRUE parallel. Paired: both arms share
# adaptive-band / universe / explore-frac / ema / group / iters / eval-seeds / --seed,
# differing ONLY in optimizer+LR (SGD-2e-3 vs AdamW-1e-5). A shared --seed makes the two
# arms' early curriculum draws identical; they diverge only as their policies diverge
# (inherent to an adaptive curriculum).
#
# Usage (on the server; RC_ROOT=/home/guest for amp_csi):
#   nohup bash scripts/adaptive_sweep.sh <run> <opt> <lr> \
#         > results/adaptive/<run>.nohup 2>&1 &
#   e.g. v4:      adaptive_sweep.sh adaptive_sgd2e3   sgd   2e-3
#        amp_csi: adaptive_sweep.sh adaptive_adamw1e5 adamw 1e-5
set -uo pipefail
ROOT="${RC_ROOT:-/home/v4}"
train="$ROOT/rl-train-t_3ed65912"
cd "$train"
run="${1:?run name required}"; opt="${2:?optimizer required}"; lr="${3:?lr required}"
base="$train/results/adaptive"; mkdir -p "$base"
out="$base/$run"; mkdir -p "$out"
slog="$base/${run}.log"; : > "$slog"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$slog"; }

# Adaptive-curriculum config (operator: band 1 7, group 8, iters 30, fixed eval seeds).
# horizon 208 / terminal_plus_hold / eta 0.1 kept from the baseline. The --adaptive-* args
# + --seed are grpo_train_loop CLIENT args; optimizer/lr/sampler/eta/grad-clip are
# TRAINER-SERVER args (set at trainer-start). universe 240 candidate seeds, explore 0.25.
COMMON="--train-skill grasp --eta 0.1 --group 8 --iters 30 --update-epochs 1 \
--clip 0.1 --hold-steps 20 --reward-variant terminal_plus_hold \
--horizon-grasp 208 --eval-n 20 --heldout-n 20 --save-videos 6 \
--adaptive-band 1 7 --adaptive-universe 240 --adaptive-explore-frac 0.25 \
--adaptive-ema 0.5 --adaptive-avoid-recent 3 --seed 12345"

log "== ADAPTIVE sweep arm $run: optimizer=$opt lr=$lr (band 1 7, universe 240, explore 0.25, ema 0.5) =="

# clean any stale client of this name + any prior trainer, then start the trainer with
# THIS arm's optimizer/lr (pi-RL sampler, eta 0.1) FRESH so both arms start from the
# identical base checkpoint.
docker rm -f "xiaomi-client-grpo-$run" >>"$slog" 2>&1 || true
bash scripts/run-train.sh trainer-stop >>"$slog" 2>&1 || true
bash scripts/run-train.sh trainer-start \
  --sampler pirl --eta 0.1 --optimizer "$opt" --lr "$lr" \
  --grad-clip 0.5 --clip 0.1 --update-epochs 1 >>"$slog" 2>&1 \
  || { log "trainer-start failed for $run"; exit 3; }
sleep 20  # model load

log "train $run ... (adaptive curriculum runs INLINE per iter; no upfront scan)"
bash scripts/run-train.sh train "$run" $COMMON >>"$out/arm.log" 2>&1
rc=$?
# run-train.sh writes results/$run ; copy the key artifacts under adaptive/$run for aggregation
if [ -d "$train/results/$run" ] && [ "$train/results/$run" != "$out" ]; then
  for f in run_summary.json train_log.jsonl eval_before.json eval_after.json \
           eval_heldout.json seed_difficulty.json; do
    cp -f "$train/results/$run/$f" "$out/$f" 2>/dev/null || true
  done
fi
log "arm $run finished rc=$rc"
bash scripts/run-train.sh trainer-stop >>"$slog" 2>&1 || true

echo "OK $(date +%s) rc=$rc" > "$base/${run}.DONE"
log "== ADAPTIVE sweep arm $run DONE (rc=$rc) =="
