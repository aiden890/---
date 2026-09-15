#!/usr/bin/env bash
# Overnight autonomous B-plan GRPO sweep (operator directive, run30).
#
# Goal: find a GRPO config whose GRASP official-success rate actually rises above base,
# with the boundary-compliance (post-success hold) reward. Honest reporting: every run's
# base-vs-trained success + post-success stay-fraction accumulated into a markdown table.
#
# Runs ONE config at a time (single RTX3090 shared with xiaomi-server). Trainer server is
# started per-config with that config's lr/kl (server-side hyperparams), the GRPO loop
# runs train+eval, results are appended, trainer stopped. A 3-strikes guard aborts the
# sweep if the same failure recurs (no infinite overnight thrash).
#
#   bash scripts/skill_sft_gate3_driver.sh  # (SFT, superseded)
#   nohup bash scripts/grpo_boundary_sweep.sh > results/grpo_sweep/driver.log 2>&1 &
set -uo pipefail
cd /home/v4/rl-train-t_3ed65912
OUT=results/grpo_sweep
mkdir -p "$OUT"
REPORT=REPORT/grpo_boundary_sweep.md
TABLE="$OUT/sweep_table.tsv"
run_train() { bash scripts/run-train.sh "$@"; }

ts() { date -u +%FT%TZ; }
log() { echo "[$(ts)] $*"; }

# --- shared eval/train budget (kept modest for cost/time; N raised only for the winner) ---
EVAL_N=12          # paired base-vs-trained eval seeds (target split)
GROUP=6            # GRPO group size (within-start advantage)
ITERS=20           # optimizer iterations per config
HOLD_STEPS=20      # post-success hold window
SEED_EVAL=5000

# Config axes: "name  lr  kl_coef  clip  group  iters  hold_stay_bonus hold_drift_penalty update_epochs"
# lr is a SERVER hyperparam (trainer-start); the rest are loop args.
CONFIGS=(
  "c1_lr1e5_kl005     1e-5  0.005 0.1 6 20 0.05 0.10 2"
  "c2_lr3e5_kl005     3e-5  0.005 0.1 6 20 0.05 0.10 2"
  "c3_lr3e6_kl002     3e-6  0.002 0.1 6 20 0.05 0.10 2"
  "c4_lr1e5_hold_hi   1e-5  0.005 0.1 6 20 0.10 0.20 2"
  "c5_lr1e5_grp8      1e-5  0.005 0.1 8 20 0.05 0.10 2"
  "c6_lr3e5_kl001     3e-5  0.001 0.2 6 25 0.05 0.15 2"
)

STRIKES=0
LAST_ERR=""
strike() {
  local err="$1"
  if [ "$err" = "$LAST_ERR" ]; then STRIKES=$((STRIKES+1)); else STRIKES=1; LAST_ERR="$err"; fi
  log "STRIKE $STRIKES ($err)"
  if [ "$STRIKES" -ge 3 ]; then
    log "ABORT: 3 consecutive same-cause failures ($err). Stopping sweep for human review."
    echo -e "\n## ABORTED $(ts)\n3 consecutive failures: $err\n" >> "$REPORT"
    run_train trainer-stop 2>/dev/null || true
    exit 3
  fi
}

wait_trainer() {
  for i in $(seq 1 60); do
    docker logs xiaomi-grpo-trainer-t_3ed65912 2>&1 | grep -q "Model loaded" && return 0
    docker logs xiaomi-grpo-trainer-t_3ed65912 2>&1 | grep -qiE "Traceback|Error|CUDA out of memory" && return 1
    sleep 5
  done
  return 1
}

# --- report header ---
if [ ! -f "$REPORT" ]; then
  {
    echo "# B-plan GRPO boundary-compliance sweep (t_e5cd5736, run30)"
    echo ""
    echo "base = pinned Xiaomi-Robotics-1-RoboCasa365 ckpt (3a6d0293), skill=GRASP, split=target,"
    echo "eval_n=$EVAL_N paired seeds (base 5000..), eta=0 deterministic eval. Reward = simulator"
    echo "milestones + boundary post-success hold (stop=+, drift=-). Honest: loss down but success"
    echo "flat is reported as flat."
    echo ""
    echo "| config | lr | kl | clip | grp | iters | base succ | trained succ | base grasp | trained grasp | hold_stay_frac | verdict |"
    echo "|---|---|---|---|---|---|---|---|---|---|---|---|"
  } > "$REPORT"
fi
: > "$TABLE"

log "=== BASE eval (once, shared across configs) ==="
# One base eval reused for all configs: skip-train, iters 0, eval only, adapter zero-init.
run_train trainer-start --optimizer adamw --lr 1e-5 --train-mode adapter_only --rank 8 >/dev/null 2>&1
if ! wait_trainer; then log "base trainer failed to load"; strike "trainer_load"; fi
run_train train grpo_base_eval --iters 0 --eval-n "$EVAL_N" --heldout-n 0 \
  --eval-split target --train-skill grasp --eval-seed-base "$SEED_EVAL" \
  > "$OUT/base_eval.log" 2>&1
run_train trainer-stop >/dev/null 2>&1
BASE_SUCC=$(python3 -c "import json;d=json.load(open('$OUT/grpo_base_eval/eval_before.json'));print(d['official_success_rate'])" 2>/dev/null || echo NA)
BASE_GRASP=$(python3 -c "import json;d=json.load(open('$OUT/grpo_base_eval/eval_before.json'));print(d['grasp_success_rate'])" 2>/dev/null || echo NA)
log "BASE succ=$BASE_SUCC grasp=$BASE_GRASP"

for spec in "${CONFIGS[@]}"; do
  read -r name lr kl clip grp iters hsb hdp ue <<< "$spec"
  log "=== CONFIG $name (lr=$lr kl=$kl clip=$clip grp=$grp iters=$iters hold_sb=$hsb hold_dp=$hdp ue=$ue) ==="
  run_train trainer-stop >/dev/null 2>&1
  sleep 2
  run_train trainer-start --optimizer adamw --lr "$lr" --train-mode adapter_only --rank 8 >/dev/null 2>&1
  if ! wait_trainer; then log "$name trainer load failed"; strike "trainer_load"; continue; fi
  run_train train "grpo_$name" --iters "$iters" --eval-n "$EVAL_N" --heldout-n 0 \
    --eval-split target --train-skill grasp --eval-seed-base "$SEED_EVAL" \
    --group "$grp" --kl-coef "$kl" --clip "$clip" --update-epochs "$ue" \
    --hold-steps "$HOLD_STEPS" --hold-stay-bonus "$hsb" --hold-drift-penalty "$hdp" \
    --ckpt-name "grpo_$name.pt" > "$OUT/train_$name.log" 2>&1
  rc=$?
  run_train trainer-stop >/dev/null 2>&1
  if [ "$rc" -ne 0 ]; then
    log "$name train rc=$rc"
    tail -5 "$OUT/train_$name.log" | sed 's/^/    /'
    if grep -qi "out of memory" "$OUT/train_$name.log"; then strike "oom"; else strike "train_rc"; fi
    echo "| $name | $lr | $kl | $clip | $grp | $iters | $BASE_SUCC | ERR | $BASE_GRASP | ERR | - | run-failed |" >> "$REPORT"
    continue
  fi
  STRIKES=0; LAST_ERR=""
  # parse trained eval + hold stats
  python3 - "$name" "$lr" "$kl" "$clip" "$grp" "$iters" "$BASE_SUCC" "$BASE_GRASP" "$OUT" "$REPORT" << 'PYEOF'
import json, sys
name, lr, kl, clip, grp, iters, bsucc, bgrasp, out, report = sys.argv[1:11]
d = f"{out}/grpo_{name}"
try:
    af = json.load(open(f"{d}/eval_after.json"))
    tsucc = af["official_success_rate"]; tgrasp = af["grasp_success_rate"]
except Exception as e:
    tsucc = tgrasp = "NA"
# mean post-success stay-fraction over the training curve (boundary compliance trend)
stay = None
try:
    rows = [json.loads(l) for l in open(f"{d}/train_log.jsonl") if l.strip()]
    fr = [r.get("mean_hold_stay_frac") for r in rows if r.get("mean_hold_stay_frac") is not None]
    if fr: stay = round(sum(fr)/len(fr), 3)
except Exception:
    pass
def fnum(x):
    try: return float(x)
    except: return None
bs, ts_ = fnum(bsucc), fnum(tsucc)
if ts_ is None or bs is None:
    verdict = "parse-err"
elif ts_ > bs:
    verdict = f"IMPROVED (+{round(ts_-bs,3)})"
elif ts_ == bs:
    verdict = "flat (=base)"
else:
    verdict = f"REGRESSED ({round(ts_-bs,3)})"
row = f"| {name} | {lr} | {kl} | {clip} | {grp} | {iters} | {bsucc} | {tsucc} | {bgrasp} | {tgrasp} | {stay} | {verdict} |"
open(report, "a").write(row + "\n")
print("ROW", row)
PYEOF
  log "$name done"
done

log "=== SWEEP COMPLETE ==="
echo -e "\n_sweep finished $(ts)_" >> "$REPORT"
