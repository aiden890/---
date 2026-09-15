# INDEX — CloseBlenderLid skill-conditioned GRPO (task t_3ed65912)

## ⚠ AUDIT INVALIDATION NOTICE (task t_37303cd3)
A code audit found that eta=0 deterministic eval ran with the trained adapter DISABLED
(`op_sample` used `skill if eta>0 else None`), so all adapter-only before/after
success-rate deltas below are NOT valid evidence of policy improvement. The GRPO
pipeline, reward layer, and checkpoints remain valid; the success deltas do not.
See REPORT/audit_fixes.md for the full gate table, fixes, and preserved
`*_invalid_pre_fix` artifacts. Any re-run for policy-improvement claims must use the
fixed eta=0 eval and paired before/after seeds, labelled post-fix.

Primary report: REPORT/arm_comparison.md · Audit report: REPORT/audit_fixes.md

## Results at a glance (N=50 eval, identical seeds 5000-5049; heldout 9000-9049)

| Arm | mode | official before→after | grasp before→after | heldout official / grasp |
|-----|------|-----------------------|--------------------|--------------------------|
| A run2 | adapter_only            | 0.20 → 0.16 | 0.10 → 0.16 | 0.08 / 0.26 |
| B run1 | adapter_plus_expert     | 0.20 → 0.20 | 0.10 → 0.14 | 0.16 / 0.20 |
| C      | adapter_plus_expert_vlm | deferred (run-budget) | | |

Bottom line: NO net official-success improvement in either pilot arm at N=50. The
deliverable that IS solid is a verified, reproducible, hardware-real GRPO pipeline +
reward layer. Limiter is sample budget (30 iters × group 4 = feasibility pilot).

## Verification gate (all pass, pre-training)
- results/... reward unit suite 19/19 (tests/test_reward_verification.py)
- results/eta_grad_verify.json  — flow-SDE eta>0 log-prob + gradient finite, ratio=1.0
- results/insim_verify.json     — reward.success == sim official_check_success (0 disagreements/120 steps)

## Artifacts (local, under rl-train-t_3ed65912/)
- REPORT/arm_comparison.md                         full report + reproduction + honest assessment
- results/armA_run2/{eval_before,eval_after,eval_heldout,run_summary}.json + eval_*/*.mp4
- results/armB_run1/{eval_before,eval_after,eval_heldout,run_summary}.json + eval_*/*.mp4
- results/out/armA_run2.pt   (7.1 MB, 216 LoRA tensors)
- results/out/armB_run1.pt   (29 MB, 216 LoRA + 11 action-expert tensors)
- Representative success clips (ARM B eval_after, official_success seeds):
  results/armB_run1/eval_after/after_seed5004_call{1,2,3}_*.mp4
  results/armB_run1/eval_after/after_seed5005_call{1,2,3}_*.mp4

## Reproduce (on v4)
cd /home/v4/rl-train-t_3ed65912
bash scripts/run-train.sh trainer-start --train-mode adapter_only --lr 2e-3 --rank 8    # ARM A
bash scripts/run-train.sh train armA_run2 --iters 30 --eval-n 50 --heldout-n 50 --group 4 --eta 0.6 \
  --seed-base 1000 --eval-seed-base 5000 --heldout-seed-base 9000 --ckpt-name armA_run2.pt
bash scripts/run-train.sh trainer-stop
# ARM B: trainer-start --train-mode adapter_plus_expert ... ; train armB_run1 ...

## Remote (v4) mirror
/home/v4/rl-train-t_3ed65912/results/{armA_run2,armB_run1,out}
Shared inference server xiaomi-server-t_460aea68 (:10086) left running untouched.
Our short-lived trainer container was stopped+removed after each arm.

## Reward-only ablation (executed)
| variant | eval N | official before→after | grasp before→after |
|---------|--------|-----------------------|--------------------|
| simulator_milestones (armA_run2)   | 50 | 0.20→0.16 | 0.10→0.16 |
| simulator_terminal_only            | 20 | 0.15→0.05 | 0.10→0.15 |
Both simulator-based; neither improved official success. sim+VLM-auxiliary variant needs a
real VLM scorer (not wired) → continuation task t_37303cd3.

## Not done (continuation task t_37303cd3, parent=this)
- sim-only vs sim+VLM-auxiliary reward ablation with a REAL VLM scorer (build + gate-verify).
- ARM C (adapter_plus_expert_vlm).
- Scaled convergence run (larger group / more iters / multi-GPU): per-iter cost ~95 s, 10.5 GB.
