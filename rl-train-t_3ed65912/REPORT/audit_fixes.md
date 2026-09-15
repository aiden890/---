# Code-audit correction report — CloseBlenderLid GRPO (task t_37303cd3)

Two operator audit rounds (base + addendum) flagged correctness issues that INVALIDATE
the earlier policy-improvement numbers and required fixes + regression gates before any
long training resumes. This documents what was invalidated, what was fixed, the evidence,
and what remains FAIL/TODO. Nothing was dressed up; unconverged/no-improvement stays that.

## Invalidated prior artifacts (preserved, not overwritten)
Renamed on v4 and locally under `results/`:
- `armC_run1_invalid_pre_fix/`             ARM C run, stopped at train iter 22/30 (before after/heldout).
- `ablation_sim_vlm_invalid_pre_fix/`      sim+VLM ablation N=50 (adapter_only) full run.
- `out/ablation_sim_vlm_invalid_pre_fix.pt` its checkpoint.
- ARM A (`armA_run2`) and the sim+VLM ablation report adapter-only before/after deltas
  that are **not valid evidence of policy improvement** because of bug #1 below (eta=0
  eval ran the BASE policy). Their pipeline/reward artifacts remain valid; the success-rate
  deltas do not.

## Gate table (before any long ARM re-run)

| # | Audit item | Status | Evidence |
|---|-----------|--------|----------|
| 1 | eta=0 eval disabled the trained adapter (`skill if eta>0 else None`) | **PASS (fixed)** | results/audit/audit_p0_verify.json → fix1_pass; d_trained=0.127 (trained adapter changes eval), d_zeroinit=0, d_none=0 (skill isolation) |
| 2 | op_load restored LoRA only, not expert/VLM `extra_trainable` | **PASS (fixed)** | audit_p0_verify.json → fix2_pass; d_restore=0.0 after corrupt+reload, n_extra_saved==n_extra_loaded==12 |
| 1b | Deterministic eval not replayable (unseeded x0 from `torch.randn_like`) | **PASS (fixed)** | eta=0 now routed through seeded flow-SDE (`op_sample`); eval derives action seed = env_seed*131+call |
| 2b | RNG re-seeded to same value every replan chunk | **PASS (fixed)** | per-chunk derived seed = seed*1_000_003 + chunk_index*9_176+1; `chunk_index` threaded from loop |
| 4a | Full simulator-state restore (not qpos-only NPZ) | **PASS** | results/audit/audit_branch_verify.json → fix4a; d_restore_qpos=0.0, all 6 predicates match, deterministic replay d=0.0 |
| 4b | Shared-prefix group branching (Z-1) | **PASS** | audit_branch_verify.json → fix4b; members start identical (d=0.0, preds identical), diverge (d=1.48) |
| 5 | Dropout disabled during rollout + logprob recompute (pi-RL) | **PASS** | server asserts `assert_no_active_dropout` at init and before every update; no active Dropout(p>0) found |
| 6 | update_epochs=1 cannot exercise clip/KL | **PASS (fixed)** | results/audit/audit_multiepoch_train_log.jsonl; epoch0 ratio==1.0 clipfrac 0, epochs1-3 ratio 0.94–1.05 clipfrac 0.47–0.75 KL 0.02–0.036; clamp-before-exp, ESS/clip-fraction/nonfinite logged |
| 10 | ARM C "VLM joint training" scope | **PASS (honest label)** | select_trainable logs groups={lora,expert_proj,vlm_slice=2560}; only `language_model.norm` slice opened — documented as a memory-bounded slice, NOT Z-1-equivalent VLM+AE joint training |

## Still FAIL / TODO (addendum items not yet gated — long ARM run NOT resumed)

| # | Item | Status | Plan |
|---|------|--------|------|
| 3 | flow_sde.py is Z-1-style fixed-noise Gaussian transition, NOT pi-RL marginal-preserving Flow-SDE | **TODO / relabel** | Chose option (b): keep the tractable Gaussian transition, remove any "marginal preserving" claim, add ODE-vs-stochastic action-dist + success checks across eta. Not a blocker for the corrected pilot, but the doc claim is being fixed. |
| 3b | structured SkillCall conditioning (RT-H/HiVLA: goal+skill+object+dest+constraints, bbox/crop) | **TODO (P1)** | `_run_one_skill` still uses fixed `SKILL_INSTRUCTION[skill]` only. Planned: pass full SkillCall + task goal into the prompt; add test that each field changes model input. |
| 4c | per-skill entry-state datasets (MOVE from post-grasp, PLACE from pre-place) actually used in `train_iteration` | **TODO (P1)** | branch-restore machinery now PROVEN (item 4). Next: build entry-state pool and have train_iteration restore it instead of resetting all skills to task-init. |
| 7 | reward milestone isolation per running skill; exclude already-successful-at-reset seeds | **TODO (P1)** | reward.py milestones currently global. Plan: gate milestones to the active skill; stratify reset-success seeds. |
| 8 | eval manifest (env seed, action-noise seed, realized asset/geom ids, randomizer values) + paired CIs + ODE/stochastic split | **TODO (P2)** | build manifest from sim; item 6 (native split honesty) folds in here. |
| 6c | full checkpoint/resume (adapter+extra+optimizer+iter+RNG+hashes) + interrupted-resume test | **TODO (P2)** | op_save/op_load cover adapter+extra; optimizer/iter/RNG/hash + resume-equality test pending. |
| 9 | action alignment invariant assert (shape/order, store native+converted) | **TODO (P2)** | convert_action verified pure copy/slice in image; add explicit invariant assert + metadata. |
| 5b (planner) | VLMPlannerStub mirrors oracle — must not be called "learned planner"; build real VLM planner | **TODO (P3)** | oracle kept for env validation only; real structured-SkillCall VLM planner is a separate build. |

## What changed (files, all inside existing dirs — no new frameworks)
- `rl-train-t_3ed65912/src/grpo_trainer_server.py`
  - #1 `op_sample`: `set_active_skill(skill)` (decoupled from eta); eta=0 routed through the
    seeded flow-SDE for replayable x0; per-(seed,chunk) derived RNG.
  - #2 `op_load`: restores `extra_trainable` (expert_proj + vlm_slice) by name, errors on any missing.
  - #5 init: force all Dropout to eval + `assert_no_active_dropout`; re-asserted each update.
  - #6 `op_update`: multi-epoch loop; clamp log-ratio before exp; logs per-epoch mean_ratio,
    clip_fraction, mean_kl, ESS, nonfinite/dropped, grad_norm; returns epoch0 vs epochLast ratio.
  - `--update-epochs` arg.
- `rl-train-t_3ed65912/src/grpo_train_loop.py`
  - eval_episode passes a deterministic per-call action seed (replayable paired before/after);
    `_run_one_skill` threads a monotonic `chunk_index`; `--update-epochs` forwarded to op=update.
- `rl-train-t_3ed65912/scripts/audit_p0_verify.py`  — #1/#2 GPU regression (scope-labelled p0_subset).
- `rl-train-t_3ed65912/scripts/audit_branch_verify.py` — #4 real simulator restore + shared-prefix branch.
- `rl-train-t_3ed65912/scripts/run-train.sh` — `audit-p0`, `audit-branch` subcommands + `--update-epochs`.

## Exact reproduction commands (on v4)
    cd /home/v4/rl-train-t_3ed65912
    # P0 adapter+checkpoint subset (own container, no trainer/shared server needed):
    bash scripts/run-train.sh audit-p0
    # item #4 simulator restore + shared-prefix branch (needs trainer up):
    bash scripts/run-train.sh trainer-start --train-mode adapter_plus_expert_vlm --lr 2e-3 --rank 8
    bash scripts/run-train.sh audit-branch --seed 5004 --group 3
    # item #6 multi-epoch optimizer smoke (ratio must leave 1 after epoch 0):
    bash scripts/run-train.sh train audit_multiepoch --iters 2 --group 4 --eta 0.6 --skip-eval \
      --update-epochs 4 --seed-base 1000 --ckpt-name audit_multiepoch.pt
    bash scripts/run-train.sh trainer-stop

## Decision on resuming long ARM runs
Pre-resume gates required by the operator (P0 adapter/checkpoint + items 1,2,4,5 + multi-epoch
optimizer smoke) all PASS. The remaining addendum items (3,3b,4c,7,8,9,5b) are P1–P3 and are
listed TODO above. Per the audit these do not all block a corrected pilot, but ANY re-run of
ARM A/B/C or the sim+VLM ablation for POLICY-IMPROVEMENT claims must (a) use the fixed eta=0
eval, (b) report paired before/after on identical env+action-noise seeds, and (c) be labelled
post-fix and kept separate from the invalid_pre_fix numbers. Shared inference server
xiaomi-server-t_460aea68 (:10086) was never stopped; our trainer container is stopped+removed.
