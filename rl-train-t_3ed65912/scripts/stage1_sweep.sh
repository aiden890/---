#!/usr/bin/env bash
# Stage-1 optimizer/LR screening for GRASP GRPO (task t_74a6ed7a), one server's arms.
# Runs its arms SEQUENTIALLY (one GPU): for each arm, (re)start the trainer with that
# arm's optimizer+LR, run a paired-seed screening train+eval, stop the trainer, next arm.
# Detached-friendly (nohup). The 2-GPU parallelism is achieved by launching this on v4 AND
# amp_csi at once with disjoint arm sets; both write into results/stage1/<run>/ so the
# finisher can aggregate all four arms.
#
# Common screening config (reference doc "Staged sweep" + task body):
#   pi-RL noise 0.1, group 8, update_epochs 1, 30 iter, clip 0.1, grad_clip 0.5,
#   PAIRED train/eval seeds (defaults shared across arms), eval 20 paired + 20 held-out,
#   reward_variant terminal_plus_hold, horizon_grasp 208.
#
# Usage (on the server; RC_ROOT set for amp_csi):
#   nohup bash scripts/stage1_sweep.sh <server_tag> <run:opt:lr> [<run:opt:lr> ...] \
#         > results/stage1/<server_tag>.nohup 2>&1 &
#   e.g. v4:      stage1_sweep.sh v4      s1_sgd2e3:sgd:2e-3   s1_adamw3e6:adamw:3e-6
#        amp_csi: stage1_sweep.sh ampcsi  s1_adamw1e5:adamw:1e-5 s1_adamw3e5:adamw:3e-5
set -uo pipefail
ROOT="${RC_ROOT:-/home/v4}"
train=$ROOT/rl-train-t_3ed65912
cd "$train"
server_tag="${1:?server_tag required}"; shift
arms=("$@")
[ "${#arms[@]}" -gt 0 ] || { echo "no arms given"; exit 2; }
base="$train/results/stage1"; mkdir -p "$base"
slog="$base/${server_tag}.log"; : > "$slog"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$slog"; }

# Common client config (identical across ALL arms -> paired comparison valid). Note:
# grad_clip/optimizer/lr/sampler/eta are TRAINER-SERVER args (set at trainer-start), NOT
# client args -- passing them to the client 'train' subcommand errors (unrecognized).
COMMON="--train-skill grasp --eta 0.1 --group 8 --iters 30 --update-epochs 1 \
--clip 0.1 --hold-steps 20 --reward-variant terminal_plus_hold \
--horizon-grasp 208 --eval-n 20 --heldout-n 20 --save-videos 6"

log "== Stage-1 sweep on $server_tag: arms=${arms[*]} =="
for spec in "${arms[@]}"; do
  run="${spec%%:*}"; rest="${spec#*:}"; opt="${rest%%:*}"; lr="${rest#*:}"
  out="$base/$run"; mkdir -p "$out"
  log "--- arm $run: optimizer=$opt lr=$lr ---"

  # clean any prior trainer, start with THIS arm's optimizer/lr (pi-RL sampler, eta 0.1)
  bash scripts/run-train.sh trainer-stop >>"$slog" 2>&1 || true
  bash scripts/run-train.sh trainer-start \
    --sampler pirl --eta 0.1 --optimizer "$opt" --lr "$lr" \
    --grad-clip 0.5 --clip 0.1 --update-epochs 1 >>"$slog" 2>&1 \
    || { log "trainer-start failed for $run"; continue; }
  sleep 20  # model load

  # screening run: results land under results/$run (run-train.sh 'train' convention),
  # then symlink into the stage1 tree for the finisher.
  log "train $run ..."
  # grad-clip is a CLIENT-unknown server arg; the client only needs COMMON. --clip is shared.
  bash scripts/run-train.sh train "$run" $COMMON >>"$out/arm.log" 2>&1
  rc=$?
  # run-train.sh writes to results/$run ; link it under stage1 for aggregation
  if [ -d "$train/results/$run" ] && [ "$train/results/$run" != "$out" ]; then
    cp -f "$train/results/$run/run_summary.json" "$out/run_summary.json" 2>/dev/null || true
    cp -f "$train/results/$run/train_log.jsonl" "$out/train_log.jsonl" 2>/dev/null || true
    cp -f "$train/results/$run/eval_before.json" "$out/eval_before.json" 2>/dev/null || true
    cp -f "$train/results/$run/eval_after.json"  "$out/eval_after.json"  2>/dev/null || true
    cp -f "$train/results/$run/eval_heldout.json" "$out/eval_heldout.json" 2>/dev/null || true
  fi
  log "arm $run finished rc=$rc"
  bash scripts/run-train.sh trainer-stop >>"$slog" 2>&1 || true
done

echo "OK $(date +%s)" > "$base/${server_tag}.DONE"
log "== Stage-1 sweep on $server_tag DONE =="
