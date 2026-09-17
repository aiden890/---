# Non-joint all-linear GRASP 30-iteration final report

Task: `t_80f3cba3`
Run: `nonjoint_abbb8c8_alllinear30`
Source commit: `abbb8c824456e75fb91274ff7abf5bbf9e9bcb01`
Model/task: Xiaomi-Robotics-1-RoboCasa365 / CloseBlenderLid GRASP
Verdict: update stability PASS; policy improvement NOT DEMONSTRATED.

## Root cause and correction

The earlier implementation exponentiated one joint log-probability aggregated over five denoise transitions, 16 executed rows, and 12 action dimensions. Small elementwise policy changes therefore accumulated across 960 terms and produced dimension-driven KL/clip saturation that barely responded to learning-rate reduction.

Commits `a078e08` and `abbb8c8` changed the pi-RL path to match RLinf's `joint_logprob=False` behavior: select one stochastic denoise transition and preserve `[L,A]` log-probabilities, ratios, clipping, KL, ESS, masks, and loss reduction elementwise. The deterministic regression includes a mutation showing that a tiny scalar log-probability delta remains stable elementwise but clips after the old synthetic joint sum.

## Stability micro gate

Exact source `abbb8c824456e75fb91274ff7abf5bbf9e9bcb01`, AdamW LR `1e-6`, group 8, update epochs 2:

- epoch-0 ratio: 1.0
- post-step KL: 0.0002145869 (gate <= 0.05)
- post-step clip fraction: 0.001798 (gate <= 0.20)
- post-step ESS: 0.999566 (gate >= 0.80)
- adapter delta L2: 0.003716
- non-finite/dropped: 0/0
- strict checkpoint mutation roundtrip: PASS; all 189 LoRA-B tensors changed the action, then 378 named tensors and the eta=0 action restored bit-exactly

This isolated the correction: the same LR that previously yielded KL 0.2903 and clip 0.7927 passed after removing the accidental joint-density aggregation.

## 30-iteration configuration

- action-expert all-linear LoRA: 189 target modules, 6,352,288 trainable parameters, rank 8, alpha 32
- pi-RL Flow-SDE, eta 0.1, one stochastic denoise transition, elementwise policy ratios
- group size 8; one group per iteration/update opportunity
- 30 rollout iterations, up to 2 optimizer epochs per trainable group
- AdamW LR 1e-6, weight decay 0.01, clip 0.2, target KL 0.05, grad clip 1.0
- horizon 208; GRASP success requires the exact 20-step hold
- reward: decayed skill terminal plus hold; milestones and approach shaping disabled
- adaptive moving band [1,7], universe 240, explore fraction 0.25, EMA 0.5, avoid-recent 3, RNG seed 12345
- paired eta=0 eval seeds 5000..5019; disjoint held-out seeds 9000..9019

This is a one-group/update stability and learning pilot, not formal Z-1-scale training. Z-1's reported global batch is much larger; multi-group production verification is owned by child task `t_eed2e2d9`.

## Measured training result

- rollout iterations: 30
- trajectories: 240
- optimizer updates: 20
- gated iterations: 10 (5 all-failure, 5 low-reward-std/all-success)
- group composition: 18 mixed, 7 all-success, 5 all-failure
- adaptive draws: 13 explore, 17 exploit
- GRASP successes: 127/240; exact-hold successes: 127/240 (equal on every iteration)
- stored trainable chunks: 1,857 across the 20 updates
- epoch-0 ratio: exactly 1.0 on every optimizer update
- non-finite/dropped ratio terms: 0/0
- target-KL early stops: 0
- peak allocated GPU memory: 10.53 GB

Post-step diagnostics over the 20 optimizer updates:

| metric | mean | min | max | gate |
|---|---:|---:|---:|---:|
| KL | 0.00025694 | 0.00020241 | 0.00031111 | <= 0.05 |
| clip fraction | 0.00226446 | 0.00158228 | 0.00287624 | <= 0.20 |
| ESS | 0.99948710 | 0.99939747 | 0.99959731 | >= 0.80 |
| grad norm | 0.70498 | 0.41016 | 2.09375 | finite |
| adapter delta L2/update | 0.00167191 | 0.00127399 | 0.00385048 | > 0 |

All stability gates passed with a large margin. The run completed, wrote `DONE`, and automatically stopped the trainer/client; v4 had no remaining containers, compute processes, or GPU allocation when checked.

## Evaluation

| split/metric | base before | trained after | delta |
|---|---:|---:|---:|
| paired GRASP exact-hold success, n=20 | 12/20 (0.60) | 12/20 (0.60) | 0.00 |
| paired official full-task success, n=20 | 3/20 (0.15) | 3/20 (0.15) | 0.00 |
| paired mean total reward | -0.713665 | -0.797970 | -0.084305 |
| held-out GRASP exact-hold success, n=20 | n/a | 11/20 (0.55) | n/a |
| held-out official full-task success, n=20 | n/a | 2/20 (0.10) | n/a |

Per-seed paired transitions were also flat in aggregate:

- GRASP: one fail->success, one success->fail, 18 unchanged
- official success: two fail->success, two success->fail, 16 unchanged

The evaluation artifact does not record completion step, so no completion-step comparison can be made from this run. This is a measurement gap, not evidence that completion timing was unchanged.

## qkv-only context

The earlier qkv-only adaptive AdamW 1e-5 run on the same nominal 5000/9000 seed bands reported paired GRASP 0.55->0.60 and held-out GRASP 0.55, with official 0.05->0.15 and held-out official 0.10. The all-linear run finished at the same 0.60 paired GRASP, 0.55 held-out GRASP, 0.15 paired official, and 0.10 held-out official, but its corrected deterministic baseline was 0.60/0.15 rather than 0.55/0.05. Therefore this is contextual comparison only, not an exact head-to-head capacity ablation. The supported conclusion is that all-linear capacity did not improve its own paired baseline.

## Artifacts and integrity

Remote canonical artifacts:

- `/home/v4/rl-train-t_3ed65912/results/nonjoint_abbb8c8_alllinear30/`
- checkpoint: `/home/v4/rl-train-t_3ed65912/results/nonjoint_abbb8c8_alllinear30/grpo_trained.pt`
- checkpoint SHA-256: `54c51e7cf8958a1643acb88adba55eacbe14195324fc1386594280c9d1849883`
- checkpoint size: 38,583,622 bytes

Text artifacts were mirrored locally to `rl-train-t_3ed65912/results/nonjoint_abbb8c8_alllinear30/`. Local/remote SHA-256 values were compared and matched for `run_summary.json`, all three eval JSON files, `train_log.jsonl`, and `seed_difficulty.json`.

W&B self-hosted run:

- `http://100.86.183.64:8080/aiden-lab-desktop/robocasa-grpo/runs/grpo-nonjoint_abbb8c8_alllinear30`
- API verification: exactly one run with this stable ID, state `finished`, last global step 29, before/after GRASP 0.60/0.60, before/after official 0.15/0.15

## Reproduction

From `/home/v4/rl-train-t_3ed65912` after deploying exact commit `abbb8c824456e75fb91274ff7abf5bbf9e9bcb01` and confirming the GPU is free:

```sh
run=nonjoint_abbb8c8_alllinear30_repro
bash scripts/run-train.sh trainer-start --optimizer adamw --lr 1e-6 --weight-decay 0.01 --train-mode adapter_only --adapter-skills grasp --lora-targets all_linear --rank 8 --alpha 32 --sampler pirl --eta 0.1 --clip 0.2 --kl-coef 0.0 --ratio-max 10 --adv-clip 3 --grad-clip 1.0 --update-epochs 2 --target-kl 0.05
bash scripts/run-train.sh train "$run" --train-skill grasp --eta 0.1 --group 8 --iters 30 --update-epochs 2 --target-kl 0.05 --clip 0.2 --kl-coef 0.0 --ratio-max 10 --adv-clip 3 --hold-steps 20 --reward-variant terminal_plus_hold --horizon-grasp 208 --eval-n 20 --heldout-n 20 --eval-seed-base 5000 --heldout-seed-base 9000 --save-videos 6 --adaptive-band 1 7 --adaptive-universe 240 --adaptive-explore-frac 0.25 --adaptive-ema 0.5 --adaptive-avoid-recent 3 --seed 12345
bash scripts/run-train.sh trainer-stop
```

Use a fresh run ID; do not overwrite the preserved run.

## Final decision

The non-joint policy-ratio correction solved the measured stability defect and made 30 iterations safe. It did not produce a GRASP or official-success gain at this scale. Do not describe this as a successful policy improvement or as formal Z-1 training. Proceed through the already-created multi-group accumulation/launcher verification task before a larger run; add completion-step logging before making timing claims.
