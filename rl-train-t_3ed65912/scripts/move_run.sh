#!/usr/bin/env bash
# MOVE-first GRPO run (operator MOVE-first direction), boundary-compliance reward.
# MOVE_HOLDING trained from a deterministic (eta=0) grasp-success entry-state so
# success -- hence the post-success hold reward -- actually fires (GRASP never did).
#
# --entry-scan-cap prefilters candidate seeds down to grasp-entry-reaching ones and
# caches them to results/<run>/entry_seeds.json (reproducible; rerun reuses cache).
# Sizes kept modest for a first real signal check:
#   eval-n 8 (paired before/after entered seeds), iters 12 (cycled over train pool),
#   group 4, scan-cap 60 per band (~17% hit -> ~8-10 entry seeds).
set -euo pipefail
cd /home/v4/rl-train-t_3ed65912
run="${1:-move_run1}"
bash scripts/run-train.sh train "$run" \
  --iters 12 --group 4 --eval-n 8 --heldout-n 0 --eval-split target \
  --train-skill move_holding --entry-from-grasp --entry-scan-cap 60 \
  --eval-seed-base 5000 --seed-base 1000 \
  --kl-coef 0.005 --clip 0.1 --update-epochs 2 \
  --hold-steps 15 --hold-stay-bonus 0.05 --hold-drift-penalty 0.10 \
  --save-videos 4 --ckpt-name "${run}.pt"
