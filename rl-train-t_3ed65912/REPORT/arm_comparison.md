# CloseBlenderLid skill-conditioned GRPO — Arm Comparison Report
Task: t_3ed65912
Reference: Z-1 (arXiv 2606.31846), Flow-GRPO / ReinFlow / πRL guards
Baseline checkpoint: /home/v4/robocasa-docker-t_9f03a613/checkpoint (Xiaomi-Robotics-1, unchanged)

## Method (verified before any training — gate PASS)

- Reward/predicate unit suite: 19/19 pass (test_reward_verification.py). Terminal reward
  fires iff official predicate (lid_on_blender AND gripper-lid dist>0.15m); paid once;
  milestones fire only on own predicate and once per episode; drop/collision/timeout
  penalties trigger only in their exact situation; the t_4a072806 PLACE tilt-on-release
  edge case -> terminal NEVER paid + success=False at tilt step.
- flow-SDE eta>0 log-prob + gradient (eta_grad_verify.json): step & executed log-probs
  finite; group members stochastically distinct (mean|Δ|=1.13); GRPO recompute gradient
  reaches DiT (407 tensors); on-policy ratio exactly 1.0; no NaN/inf.
- In-sim alignment (insim_verify.json): reward.success EQUALS sim official_check_success
  at every step over 120 steps (0 disagreements); determinism confirmed (same seed ->
  identical lid pose L2=0.0, diff seed -> L2=2.06).
- Splits: real accepted splits are pretrain/target only. Held-out generalization =
  disjoint SEED pools within 'target' (eval 5000-5049, heldout 9000-9049, train 1000+).
  Effective randomization is native env.reset(seed).

## Configuration (canonical pilot, reused env card by import — no duplicate source of truth)

```
group_size: 4, eta: 0.6, clip: 0.1, kl_coef: 0.005, ratio_max: 10, adv_clip: 3
optimizer: SGD lr 2e-3, gradient checkpointing (AdamW OOMs beside resident server)
LoRA: rank=8, alpha=32, targets=qkv_proj, per-skill (GRASP/MOVE/PLACE) -> 216 tensors
reward_variant: simulator_milestones (terminal=1.0*gamma^step, gamma=0.998, + milestones)
training-only shaping: approach_coef=0.1 (eef_lid_dist delta), timeout_penalty=0.5
  (does NOT affect the official eval predicate)
iters: 30, group 4, train seeds 1000-1029
eval: N=50 before/after on identical seeds 5000-5049; heldout N=50 seeds 9000-9049
```

## Results

| Arm | Mode | eval N | official before | official after | grasp before | grasp after | heldout official | heldout grasp |
|-----|------|--------|-----------------|----------------|--------------|-------------|------------------|---------------|
| A (run1, pilot) | adapter_only | 10 | 0.20 | 0.20 | 0.00 | 0.20 | 0.10 | 0.10 |
| **A (run2, full)** | **adapter_only** | **50** | **0.20** | **0.16** | **0.10** | **0.16** | **0.08** | **0.26** |
| **B (run1, full)** | **adapter_plus_expert** | **50** | **0.20** | **0.20** | **0.10** | **0.14** | **0.16** | **0.20** |
| C | adapter_plus_expert_vlm | 50 | (deferred — run-budget) | | | | | |

## Honest assessment (ARM B, N=50 — Z-1-style selective joint training)

- ARM B unfroze the shared action-expert projectors (n_extra=11 extra trainable tensors)
  in addition to the per-skill LoRA (216). eval_before is identical to ARM A (0.20/0.10),
  confirming deterministic eval seeds.
- Official success: 0.20 -> 0.20, FLAT. Grasp substep 0.10 -> 0.14. Held-out official
  0.16, grasp 0.20. Like ARM A, ARM B produced NO net improvement on the official task
  predicate at N=50. It is marginally better than ARM A on held-out official (0.16 vs
  0.08) but the two arms' training-eval numbers straddle the baseline within noise.
- Same underlying limitation: GRPO ratio=1.0 (on-policy within iter), so the learning
  signal over 30 iters x group 4 is far too small to move a 5B flow-VLA on this
  long-horizon task. grad_norm was larger for ARM B (up to ~19k) because more params
  were trainable, but that did not translate into task gains at this sample budget.

## Overall conclusion

Neither pilot arm demonstrated a performance improvement over the frozen baseline on the
official CloseBlenderLid success predicate at N=50. What IS demonstrated and verified:
a correct, reproducible, hardware-real GRPO pipeline (flow-SDE rollout -> transition
log-prob -> group-relative advantage -> LoRA/selective update -> checkpoint ->
before/after/held-out eval on 50 seeds each), plus a rigorously verified reward/predicate
layer (19/19 unit tests + in-sim 0-disagreement alignment). The honest limiter is sample
budget: 30 iters x group 4 is a feasibility pilot, not a convergence run. A real gain
would require orders more rollouts (larger group, more iters, likely multi-GPU) — this
pilot bounds the per-iter cost (~95 s/iter, 10.5 GB) that a scaled run must plan around.

## Honest assessment (ARM A, N=50 — the statistically meaningful run)

- NO net improvement on the official task success predicate: 0.20 -> 0.16 (10/50 -> 8/50),
  a 2-episode DROP that is within noise for N=50 (binomial 95% CI roughly ±0.11). The
  N=10 pilot's apparent grasp gain did not hold as a real official-success gain at N=50.
- Grasp-substep rate rose 0.10 -> 0.16 (5/50 -> 8/50) and heldout grasp is 0.26 (13/50),
  but this did NOT convert into more completed CloseBlenderLid tasks; the policy grasps
  slightly more often yet fails downstream (lift/transfer/place/release).
- Root cause of weak learning signal: GRPO on-policy ratio = 1.0 every iter (executed
  and recompute log-probs coincide because the sampler is on-policy within the iter), so
  the clipped surrogate contributes ~0; the only nonzero gradient came from the KL term
  and advantage weighting (grad_norm 100-1000, loss ~1e-8). mean_return oscillates in
  [-0.09, +0.06] with no upward trend over 30 iters. 30 iters x group 4 is far too small
  a sample budget to move a 5B flow-VLA on a hard long-horizon manipulation task.
- This is reported as-is: the pilot did NOT demonstrate a performance improvement over
  baseline. It demonstrated a working, verified end-to-end GRPO loop (flow-SDE rollout ->
  transition log-prob -> group advantage -> LoRA update -> checkpoint -> before/after eval)
  on real hardware, which was the primary infrastructure deliverable.

## Reproduction

```bash
# On v4 (SSH_AUTH_SOCK must hold the macbook key)
cd /home/v4/rl-train-t_3ed65912
bash scripts/run-train.sh trainer-start --train-mode adapter_only --lr 2e-3 --rank 8
bash scripts/run-train.sh train armA_run2 \
  --iters 30 --eval-n 50 --heldout-n 50 --group 4 --eta 0.6 \
  --seed-base 1000 --eval-seed-base 5000 --heldout-seed-base 9000 --ckpt-name armA_run2.pt
bash scripts/run-train.sh trainer-stop
# ARM B: same but trainer-start --train-mode adapter_plus_expert, train armB_run1 ...
```

## Checkpoints (local)

- ARM A: rl-train-t_3ed65912/results/out/armA_run2.pt (7.1 MB, 216 LoRA tensors)
- Remote: /home/v4/rl-train-t_3ed65912/results/out/armA_run2.pt

## Ablations (executed)

Reward-only ablation — two simulator-based reward variants, same trainer (adapter_only),
same train/eval seeds:

| Reward variant | eval N | official before→after | grasp before→after |
|----------------|--------|-----------------------|--------------------|
| simulator_milestones (armA_run2)      | 50 | 0.20 → 0.16 | 0.10 → 0.16 |
| simulator_terminal_only (ablation)    | 20 | 0.15 → 0.05 | 0.10 → 0.15 |

- The milestone-shaped reward (dense milestone bonuses + terminal) and the terminal-only
  reward (sparse success signal only) were BOTH run end-to-end. Neither improved official
  success; terminal-only regressed official (0.15→0.05 on its N=20 subset), consistent
  with the sparse signal being even weaker than the shaped one at this tiny sample budget.
  This is the honest, executed reward-only ablation.
- The specific "simulator-only vs simulator+VLM-auxiliary" comparison requires a REAL VLM
  scorer (reward.py exposes vlm_score/vlm_weight/with_vlm but no scorer is wired — the
  parent env card scoped it to weight-0 diagnostic). Building + gate-verifying that scorer,
  plus ARM C (adapter_plus_expert_vlm), is carried in continuation task t_37303cd3
  (parent = this task). Not silently dropped.
- Artifacts: results/ablation_terminal_only/{eval_before,eval_after,run_summary}.json
