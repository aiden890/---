# Corrected GRPO correctness micro — final report

Task: `t_2bb64447`
Correctness base commit: `be7df4f537b4b35a921c55d5d9e46cb65eca45b0`
Unique-checkpoint launcher fix: `3154248851cceb3703b9b60931ce1e1627282bbd`
Checkpoint RNG restore fix: `6fc53e6d121352b6c41f904c93da05fae07f2ee7`
GPU run: `corrected_exact_3154248_micro_lr1e6`
Classification: correctness/feasibility micro only; `--skip-eval`; no policy-improvement claim.

## Source and deployment gate

- v4 train and env manifests both recorded commit `3154248851cceb3703b9b60931ce1e1627282bbd`, `dirty=false`.
- Every file in both manifests was re-hashed on v4 after deployment: zero mismatches.
- Trainer-recorded source manifest SHA-256: `06679ca1cf5a74e3c8de860a4242f4a91f4f004dd9533b93e3133c573560cbb6`.
- The client and trainer shared the server-visible unique output path `/train/results/corrected_exact_3154248_micro_lr1e6`.
- The checkpoint was written inside that run directory, not the old shared `results/out/` path.

The first exact-`be7df4f` launch exposed the remaining launcher bug: `run-train.sh` passed `--out /out`, so the trainer derived `/train/results/out/grpo_trained.pt`. The run was retained as `corrected_exact_be7_micro`; it was not treated as the final unique-output gate. Commit `3154248` fixes the launcher, and `test_run_train_checkpoint_path.py` includes a legacy-path mutation that must fail.

## GPU micro configuration

- Xiaomi-Robotics-1-RoboCasa365, CloseBlenderLid / GRASP
- action-expert all-linear LoRA: 189 target modules, 6,352,288 trainable parameters, rank 8, alpha 32
- piRL Flow-SDE, eta 0.1, five denoise steps
- group size 8; up to two groups to obtain at least one trainable mixed group
- horizon 208; post-success hold 20 steps
- reward: decayed skill terminal plus hold; milestones and approach shaping disabled
- AdamW, LR `1e-6`, weight decay 0.01, two PPO epochs, clip 0.2, target-KL 0.05
- one optimizer update; evaluation skipped

## Observed GPU result

- Group 0: all-success 8/8, correctly gated for zero reward variance, 75 stored chunks, 0 trainable chunks.
- Group 1: mixed 7/8, 82 stored/trainable chunks.
- Aggregate: 15/16 GRASP successes; all 15 successful trajectories held the predicate through the exact 20-step window.
- Reward decomposition for successes was explicit and sum-consistent: skill terminal `0.982143` or `0.984112` according to completion chunk, hold-stay `1.0`, all other components zero. The failure received timeout `-0.5`.
- Epoch 0 correctness probe: mean ratio `1.0`, KL `0.0`, clip fraction `0.0`, grad norm `296.0`.
- Adapter changed: L2 delta `0.00195697`, Linf delta `1.01328e-6`.
- Post-step: mean ratio `0.897178`, KL `0.290280`, clip fraction `0.792683`, ESS `0.484861`.
- Non-finite chunks: 0; dropped chunks: 0; peak allocated GPU memory: 10.52 GB; no OOM.
- Collection time: 418.191 s; optimizer time: 271.049 s.
- Append-only progress JSONL contains both completed groups before the optimizer result.
- Client exited and `run_summary.json`, `train_log.jsonl`, and the named checkpoint were written.

## Checkpoint gate

The first fresh-trainer load of the real checkpoint failed with `RNG state must be a torch.ByteTensor`: `torch.load(..., map_location=cuda)` had moved saved CPU RNG tensors to CUDA before `torch.set_rng_state`. Commit `6fc53e6` normalizes both the CPU RNG tensor and each saved CUDA RNG tensor to CPU before restore, with a regression/mutation test. After deploying `6fc53e6`, a fresh trainer with the same strict named layout loaded:

`/train/results/corrected_exact_3154248_micro_lr1e6/grpo_trained.pt`

The post-fix load returned schema version 2, 378 named LoRA tensors, 0 extra tensors, adapter skill `grasp`, the 189 target names, rank/alpha, optimizer/RNG state metadata, update index 1, base model, sampler/config, exact source commit, and source manifest hash. The checkpoint correctly retains its generating source commit `3154248`; the loader fix is `6fc53e6`. Missing/unexpected/shape mismatch checks remain fail-fast in the schema tests.

## Stability verdict

Functional correctness gate: PASS.

Production PPO stability gate: FAIL. The target-KL early-stop path fired, but it can only stop after observing the first changed-policy epoch. LR `1e-5` produced post-step KL 0.253/clip 0.756; reducing LR tenfold to `1e-6` still produced KL 0.290/clip 0.793. Therefore `1e-6..1e-5` is an empirically rejected region, not a production range. No 30-update run was started. Further scaling is blocked until the action/log-prob sensitivity is diagnosed; blindly lowering LR or increasing epochs is not justified by these two points.

## Verification rerun

Central/std-lib checks rerun:

- reward verification: 14 groups, 0 failed
- training correctness: pass
- update-batch mean-gradient tests: pass
- checkpoint schema: pass
- all-linear smoke gate unit test: pass
- source-manifest tests: pass
- unique checkpoint path baseline + mutation: 2/2 pass

Inside the real trainer image:

- LoRA isolation/gradient tests: all pass
- advantage baseline checks + mutations: 7/7 bugs caught
- training correctness, update batch, checkpoint schema, smoke gate: pass
- fixed-noise Flow-SDE: 5/5 pass
- piRL CPU math gates: all pass; RLinf reference worst difference 0; rollout/recompute ratio approximately 1

## Preserved artifacts

- `/home/v4/rl-train-t_3ed65912/results/alllinear_grasp_smoke2/` — pre-fix 0/2 diagnostic
- `/home/v4/rl-train-t_3ed65912/results/corrected_exact_be7_micro/` — exact be7 functional run with rejected shared checkpoint path and KL 0.253
- `/home/v4/rl-train-t_3ed65912/results/corrected_exact_3154248_micro_lr1e6/` — final unique-output correctness micro

W&B did not mirror `alllinear_grasp_smoke2` or these corrected runs because the existing mirror hard-codes older arms. Follow-up card `t_60a33390` owns run discovery; this gap does not change the recorded file artifacts above.
