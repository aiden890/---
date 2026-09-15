#!/usr/bin/env bash
# MOVE-first smoke test (operator MOVE-first direction).
# Verifies: (1) deterministic eta=0 grasp reaches the MOVE entry-state,
#           (2) MOVE runs from that restored state,
#           (3) MOVE success fires (n_success_hold > 0) so the boundary-hold
#               reward actually gets a signal -- the thing GRASP never had.
# Small/cheap: 3 train iters, group 4, 6 eval seeds, no heldout.
set -euo pipefail
cd /home/v4/rl-train-t_3ed65912
bash scripts/run-train.sh train move_smoke \
  --iters 3 --group 4 --eval-n 6 --heldout-n 0 --eval-split target \
  --train-skill move_holding --entry-from-grasp \
  --eval-seed-base 5000 --seed-base 1000 \
  --kl-coef 0.005 --clip 0.1 --update-epochs 2 \
  --hold-steps 15 --hold-stay-bonus 0.05 --hold-drift-penalty 0.10 \
  --save-videos 2 --ckpt-name move_smoke.pt
