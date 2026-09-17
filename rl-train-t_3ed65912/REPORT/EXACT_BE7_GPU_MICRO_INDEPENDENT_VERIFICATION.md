# Exact-source GPU micro — independent verification

Task: `t_ebd5d630`
Scope: independent post-fix evidence review and strict checkpoint reload; no implementation changes and no new optimizer update.

## Verdict

- Exact `be7df4f537b4b35a921c55d5d9e46cb65eca45b0` was deployed for `corrected_exact_be7_micro`; its `run_summary.json` records one update and that exact source commit. That run exposed the shared `/train/results/out/grpo_trained.pt` launcher-path defect, so it is diagnostic rather than the final gate.
- The final accepted micro, `corrected_exact_3154248_micro_lr1e6`, used clean commit `3154248851cceb3703b9b60931ce1e1627282bbd`, which contains `be7df4f` and adds the run-unique checkpoint-path fix. Commit ancestry was verified locally.
- Functional forward/backward/update and checkpoint gates pass. Production stability fails because the first changed-policy epoch overshot the target KL and clipped heavily. No 30-update continuation was started.

## Independently checked evidence

### Source provenance

The final run embeds complete train and env source manifests:

- train: commit `3154248851cceb3703b9b60931ce1e1627282bbd`, `dirty=false`, 55 files
- env: commit `3154248851cceb3703b9b60931ce1e1627282bbd`, `dirty=false`, 22 files
- trainer-recorded train manifest SHA-256: `06679ca1cf5a74e3c8de860a4242f4a91f4f004dd9533b93e3133c573560cbb6`

Each embedded file digest was independently compared with `git archive 3154248`; mismatches were 0/55 for train and 0/22 for env. The live v4 deployment has since advanced to another clean commit, so provenance for the historical run is established from its immutable embedded manifests rather than the current server tree.

### GPU update artifact

Remote artifact: `/home/v4/rl-train-t_3ed65912/results/corrected_exact_3154248_micro_lr1e6/`

- one optimizer update; 2 groups, 16 trajectories, 82 trainable chunks
- epoch-0 ratio `1.0`, KL `0.0`, clip fraction `0.0`, grad norm `296.0`
- adapter delta L2 `0.0019569704309105873`, Linf `1.0132789611816406e-6`
- post-step ratio `0.897177771961513`, KL `0.2902799061078548`, clip fraction `0.7926829268292683`, ESS `0.48486111934618054`
- post-step nonfinite `0`, dropped `0`; peak allocated memory `10.52 GiB`; no OOM
- group progress is append-only and contains the all-success gated group followed by the mixed 7/8 group

### Strict checkpoint reload

A fresh real Xiaomi trainer was started on v4 with the matching all-linear GRASP LoRA layout, then the preserved checkpoint was loaded through the trainer's strict `op_load` RPC. The live response passed with:

- schema version `2`
- 378 named LoRA tensors
- 0 extra tensors
- 189 target modules, rank 8, alpha 32
- update index `1`
- checkpoint source commit `3154248851cceb3703b9b60931ce1e1627282bbd`
- checkpoint source-manifest hash `06679ca1cf5a74e3c8de860a4242f4a91f4f004dd9533b93e3133c573560cbb6`

The load succeeded after restoring optimizer and RNG state; the previous CUDA-vs-CPU RNG-state failure did not recur. The verification trainer was stopped afterward. v4 ended with no active containers and no GPU compute process.

### No long run

A programmatic scan of v4 result summaries found no run from source commits `be7df4f` or `3154248` with `iters >= 30`. The only completed exact-source summaries are one-update micros. No container or GPU process remained active after verification.

### Tracking publication

The live port-8899 rendered page for `EXP-20260917-Z1-EXPERT-SMOKE` explicitly shows:

- pre-fix diagnostic classification
- `0/2` completed optimizer updates
- no checkpoint/summary/final evaluation for that pre-fix run
- `W&B mirror` does not include `alllinear_grasp_smoke2`
- 30-update continuation prohibited pending the corrected gate

The separate corrected experiment entry records functional correctness PASS, production stability FAIL, and `continue_30_updates=false`.

This verifies the requested point-in-time tracking statement. Concurrent follow-up `t_60a33390` is actively adding automatic W&B run discovery, so the statement may become historical once that separate observability task completes; it does not alter the GPU micro verdict.

## Local verification rerun

Passed:

- `tests/test_checkpoint_schema.py`
- `tests/test_run_train_checkpoint_path.py` (baseline plus mutation)
- `tests/test_training_correctness.py`
- `tests/test_update_batch.py`

Final decision: capacity/correctness PASS; production stability FAIL; no performance claim and no 30-update run.
