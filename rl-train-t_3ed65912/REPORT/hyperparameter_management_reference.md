# RL Hyperparameter Management Reference

Status: REFERENCE ONLY. Do not launch, resume, or modify training from this document until the operator explicitly instructs it.

## Audited run

Run: grasp_h208_pirl_eta01

- skill: GRASP_OBJECT from reset
- sampler: faithful piRL Flow-SDE, noise level 0.1
- horizon: 208
- iterations: 20
- group size: 4
- optimizer epochs: 1
- optimizer: SGD
- learning rate: 2e-3
- gradient clip: 0.5
- adapter: rank-8 per-skill LoRA on DiT qkv_proj, about 3.54M trainable parameters
- reward label in config: simulator_terminal_only
- actual reward: terminal success plus 20-step post-success hold shaping
- data volume: 80 trajectories and 918 stored action chunks
- evaluation: 12 paired evaluation seeds and 12 held-out seeds

Observed:
- official full-task success: 3/12 -> 3/12
- GRASP success: 8/12 -> 9/12
- held-out GRASP success: 7/12
- total successful hold trajectories during training: 55/80
- group composition: 10 all-success, 7 mixed success/failure, 3 all-failure
- 12/20 groups had reward std below 0.05; 9/20 were below 0.001
- epoch 0 ratio=1 and KL=0 are expected because update_epochs=1 computes metrics before the optimizer step
- raw gradient norm mean 1877.6, max 4288, then globally clipped to 0.5

Interpretation:
- The pipeline and piRL sampler operated correctly.
- "No behavioral change because LR was weak" is not established.
- Full-task success is not the primary metric for a GRASP-only update.
- n_success_hold=55 does not imply 55 informative comparisons. Many all-success groups had nearly identical returns, and standardizing a tiny std can amplify noise and assign negative advantage to valid successes.
- n=12 evaluation is a smoke test, not enough for a reliable improvement claim.

## Required measurement fixes before stronger training

1. Log the complete optimizer configuration in every run summary: optimizer, LR, schedule, grad clip, weight decay, LoRA rank/alpha/targets, trainable parameter count, checkpoint hash, and code revision.
2. With update_epochs=1, recompute ratio, approximate KL, clip fraction, ESS, and adapter parameter delta after the optimizer step. Pre-step ratio=1 and KL=0 are correctness checks, not update-strength diagnostics.
3. Use GRASP success as the primary metric for GRASP training. Keep official CloseBlenderLid success as a secondary end-to-end metric.
4. Report paired per-seed changes and confidence intervals. Use at least 20 paired seeds for screening and at least 50 evaluation plus 50 held-out seeds for final decisions.
5. Rename the current reward variant terminal_plus_hold. If a pure terminal-only ablation is required, set all hold reward weights to zero while retaining hold as a verifier condition.
6. Log group outcome composition and low-reward-variance fraction.

## Advantage handling before a larger sweep

- Mixed success/failure group: use group-relative normalized advantage.
- All-success group: use Z-1-style completion-aware, non-negative minimum-baseline advantage; do not mark slower successful trajectories as negative solely because the group mean is higher.
- All-failure or effectively constant-reward group: skip the update unless a separately approved shaping signal provides meaningful ordering.
- Add a minimum reward-std gate; choose the threshold from observed reward scale and report how many groups are filtered.
- Keep reward-design experiments separate from optimizer hyperparameter experiments.

## Staged sweep

Keep piRL noise level fixed at 0.1 because the completed G5 sweep showed monotonic success degradation above 0.1.

Stage 1: optimizer/LR screening
- common paired train/eval seeds
- group size 8
- update_epochs 1
- 20-30 iterations per arm
- control: SGD 2e-3
- AdamW candidates: 3e-6, 1e-5, 3e-5
- keep clip=0.1 and grad_clip=0.5 fixed

Stage 2: reuse control
- choose the best Stage-1 optimizer/LR using paired GRASP success plus post-step ratio/KL/clip diagnostics
- compare update_epochs 1 versus 2
- compare clip 0.1 versus 0.2 only if post-step ratios are stable
- do not use update_epochs 4 until two-epoch clip fraction and KL are acceptable

Stage 3: longer run
- run the selected configuration for 100 iterations
- checkpoint and evaluate every 20 iterations
- early-stop on sustained held-out regression, non-finite values, sample dropping, excessive clipping, KL growth, or adapter norm explosion

Suggested initial candidate:
- skill GRASP_OBJECT
- piRL noise 0.1
- group 8
- AdamW LR 5e-6 or 1e-5
- update_epochs 1
- clip 0.1
- grad_clip 0.5
- 30 screening iterations, then 100 only after passing gates

## Reference configurations

RLinf OpenPI GRPO configurations use group_size=8, AdamW LR around 5e-6, clip ratio 0.2, grad clip 1.0, and update_epoch 1 or 2 depending on the task/model. These values are reference points rather than drop-in settings because Xiaomi MiBoT, adapter-only training, action dimensions, and rollout batch sizes differ.

Primary references:
- piRL: https://arxiv.org/abs/2510.25889
- Z-1: https://arxiv.org/abs/2606.31846
- RLinf pi0.5 GRPO config: https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/libero_spatial_grpo_openpi_pi05.yaml
- RLinf pi0 GRPO config: https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/libero_goal_grpo_openpi.yaml
