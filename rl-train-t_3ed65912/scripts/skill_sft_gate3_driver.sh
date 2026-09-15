#!/usr/bin/env bash
# Gate-3 driver (run30, operator decision A): given a trainer already started with the
# desired recipe (anchor-coef / expert-lr / lr / alpha), train the GRASP adapter for one
# epoch and run the deterministic eta=0 rollout gate (12 target-split seeds, base vs
# trained, paired). Writes a compact PASS/FAIL summary. Intended to be launched detached
# (nohup ... &) so ssh round-trips don't have to babysit it.
#   bash scripts/skill_sft_gate3_driver.sh <tag>
set -euo pipefail
cd /home/v4/rl-train-t_3ed65912
tag="${1:?usage: skill_sft_gate3_driver.sh <tag>}"
ck="/out/grasp_${tag}.pt"
log="results/skill_sft/gate3_${tag}.log"
mkdir -p results/skill_sft
{
  echo "[driver] $(date -u +%FT%TZ) tag=$tag waiting for trainer ready"
  for i in $(seq 1 60); do
    docker logs xiaomi-grpo-trainer-t_3ed65912 2>&1 | grep -q "Model loaded" && { echo "[driver] trainer ready"; break; }
    sleep 5
  done
  echo "[driver] === TRAIN GRASP adapter (1 epoch) ==="
  bash scripts/run-train.sh sft "train_${tag}" --op train --skill GRASP_HANDLE \
    --arm nl_plus_skill_id --epochs 1 --ckpt "$ck" 2>&1 | grep -E "saved|it[0-9]{3}|Error|Trace" | tail -6
  echo "[driver] === GATE 3 rollout (12 seeds, base vs trained) ==="
  bash scripts/run-train.sh train "sft_rollout_gate_${tag}" --iters 0 --eval-n 12 \
    --heldout-n 0 --eval-split target --train-skill grasp \
    --load-ckpt "/train/results/skill_sft/ckpt/grasp_${tag}.pt" 2>&1 | grep -E "before_\]|after_\]|Error|Trace" | tail -4
  echo "[driver] === RESULT ==="
  python3 - "$tag" << 'EOF'
import json, sys
tag = sys.argv[1]
base = f"results/sft_rollout_gate_{tag}"
b = json.load(open(f"{base}/eval_before.json"))
a = json.load(open(f"{base}/eval_after.json"))
def rate(d, key):
    eps = d.get("episodes", [])
    return sum(1 for e in eps if e.get(key)), len(eps)
bg, n = rate(b, "grasp_success"); ag, _ = rate(a, "grasp_success")
bo, _ = rate(b, "official_success"); ao, _ = rate(a, "official_success")
verdict = "PASS" if ag > bg else ("FAIL(collapse)" if ag < bg else "FAIL(no-op/=base)")
print(json.dumps({"tag": tag, "n": n,
                  "base_grasp": bg, "trained_grasp": ag,
                  "base_official": bo, "trained_official": ao,
                  "gate3": verdict}))
EOF
  echo "[driver] $(date -u +%FT%TZ) done"
} > "$log" 2>&1
