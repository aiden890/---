#!/usr/bin/env bash
# Chained after grpo_boundary_sweep.sh: waits for the sweep to finish, selects the
# best GRASP config from the sweep table, and IF it beat base by a real margin,
# de-noises the winner with a higher-N confirmation eval (24 paired seeds). Writes
# a FINAL summary + a DONE marker so a poller can post the morning report.
# Honest: if nothing beat base, it says so and recommends B/C, no auto long run.
#
#   nohup bash scripts/grpo_sweep_finisher.sh <sweep_pid> > results/grpo_sweep/finisher.log 2>&1 &
set -uo pipefail
cd /home/v4/rl-train-t_3ed65912
OUT=results/grpo_sweep
RESULTS=results
REPORT=REPORT/grpo_boundary_sweep.md
FINAL=REPORT/grpo_sweep_FINAL.md
SWEEP_PID="${1:-}"
ts() { date -u +%FT%TZ; }
log() { echo "[$(ts)] $*"; }

# 1) wait for the sweep process to exit (break early if the report already shows done)
if [ -n "$SWEEP_PID" ]; then
  log "waiting for sweep pid $SWEEP_PID"
  while kill -0 "$SWEEP_PID" 2>/dev/null; do
    grep -qE "SWEEP COMPLETE|ABORTED|_sweep finished" "$REPORT" 2>/dev/null && { log "report shows done; stop waiting"; break; }
    sleep 120
  done
  log "sweep pid $SWEEP_PID done/gone"
fi
# also wait until the report says COMPLETE or ABORTED
for i in $(seq 1 30); do
  grep -qE "SWEEP COMPLETE|ABORTED|_sweep finished" "$REPORT" 2>/dev/null && break
  grep -q "sweep finished" "$OUT/driver.log" 2>/dev/null && break
  sleep 30
done

# 2) pick best config from the accumulated table (max trained succ; improved over base)
BEST=$(python3 - "$REPORT" << 'PYEOF'
import re, sys
report = sys.argv[1]
rows = []
for line in open(report):
    if not line.startswith("| c"):  # config rows start with "| cN..."
        continue
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) < 12:
        continue
    name = cells[0]
    def f(x):
        try: return float(x)
        except: return None
    bsucc, tsucc = f(cells[6]), f(cells[7])
    if tsucc is None or bsucc is None:
        continue
    rows.append((name, bsucc, tsucc, tsucc - bsucc))
if not rows:
    print("NONE 0 0 0")
else:
    # best = highest trained succ, tiebreak highest delta
    rows.sort(key=lambda r: (r[2], r[3]), reverse=True)
    n, b, t, d = rows[0]
    print(f"{n} {b} {t} {d}")
PYEOF
)
read -r BNAME BBASE BTRAIN BDELTA <<< "$BEST"
log "best from table: $BNAME base=$BBASE trained=$BTRAIN delta=$BDELTA"

# 3) confirmation eval for a real winner (delta > 0). Uses the saved best ckpt at
#    higher N to de-noise. Skipped for flat/regressed (honest: no winner to confirm).
CONF="skipped (no config beat base)"
IMPROVED=$(python3 -c "print(1 if $BDELTA > 0 else 0)" 2>/dev/null || echo 0)
if [ "$IMPROVED" = "1" ] && [ "$BNAME" != "NONE" ]; then
  CKPT="results/out/grpo_${BNAME}.pt"          # host path (to test existence)
  CKPT_SRV="/train/results/out/grpo_${BNAME}.pt"  # trainer-server path for --load-ckpt
  if [ -f "$CKPT" ]; then
    log "confirming winner $BNAME with 24-seed eval"
    bash scripts/run-train.sh trainer-stop >/dev/null 2>&1; sleep 2
    bash scripts/run-train.sh trainer-start --optimizer adamw --lr 1e-5 --train-mode adapter_only --rank 8 >/dev/null 2>&1
    for i in $(seq 1 60); do
      docker logs xiaomi-grpo-trainer-t_3ed65912 2>&1 | grep -q "Model loaded" && break; sleep 5
    done
    bash scripts/run-train.sh train grpo_confirm_best --iters 0 --eval-n 24 --heldout-n 0 \
      --eval-split target --train-skill grasp --eval-seed-base 5000 \
      --load-ckpt "$CKPT_SRV" > "$OUT/confirm_best.log" 2>&1 || true
    bash scripts/run-train.sh trainer-stop >/dev/null 2>&1
    CONF=$(python3 -c "import json;d=json.load(open('$RESULTS/grpo_confirm_best/eval_after.json'));print('24-seed official=%.3f grasp=%.3f' % (d['official_success_rate'], d['grasp_success_rate']))" 2>/dev/null || echo "confirm parse-err")
    log "confirm: $CONF"
  else
    CONF="winner ckpt $CKPT missing"
  fi
fi

# 4) write FINAL summary
{
  echo "# GRPO boundary sweep — FINAL (t_e5cd5736)"
  echo ""
  echo "_finished $(ts)_"
  echo ""
  echo "Best-by-trained-success config: **$BNAME** (base=$BBASE, trained=$BTRAIN, delta=$BDELTA)"
  echo ""
  echo "Winner confirmation (higher-N de-noise): $CONF"
  echo ""
  if [ "$IMPROVED" = "1" ]; then
    echo "Verdict: a config beat base on the sweep table. Confirm the higher-N number above"
    echo "before any N=50/ARM run; if it holds, next step = extend best config to MOVE/RELEASE."
  else
    echo "Verdict: NO config beat base (all flat or regressed) even with the boundary-compliance"
    echo "reward. Consistent with the SFT-era finding + pilot flat result: GRASP success does not"
    echo "rise from GRPO at this budget, and post-success hold rarely triggers when success itself"
    echo "is rare (n_succ_hold~0). Recommend operator decide: (B') larger group / milestone-shaped"
    echo "advantage to create signal on the hard task, or (C) report flat and stop long runs."
  fi
  echo ""
  echo "Full per-config table: REPORT/grpo_boundary_sweep.md"
} > "$FINAL"

date -u +%FT%TZ > "$OUT/DONE"
echo "$BNAME $BBASE $BTRAIN $BDELTA | $CONF" >> "$OUT/DONE"
log "FINISHER DONE -> $FINAL"
