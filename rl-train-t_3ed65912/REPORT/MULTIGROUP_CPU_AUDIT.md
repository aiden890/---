# Multi-group CPU and mutation audit

Status: CPU/static gate passed; pending GPU gate.

Purpose: verify the multi-group collection/update path before task `t_eed2e2d9` spends v4 GPU time. No deploy, remote container, trainer restart, or GPU test was performed.

## Findings and fixes

- Replaced unchecked `dict.update()` aggregation with `BatchAccumulator`, which rejects duplicate env seeds, duplicate trajectory IDs, stale/missing trainer-store trajectories, and zero-chunk trajectories.
- Centralized whole-group collection in `execute_batched_update()`. It enforces both floors, honors the max-group cap, invokes exactly one logical trainer update RPC per collected batch, and clears the trainer store on every exception path. That RPC runs the configured two optimizer epochs; it is not claimed as one `opt.step()`.
- Zero-advantage groups remain represented in the complete batch/store but contribute zero trainable chunks. Group-local advantages are preserved verbatim; there is no batch-global renormalization.
- Moved the existing `AdaptiveCurriculum` into a reusable module and integrated it per group in batch mode. Each group receives a distinct seed and its observed `n_success_group` updates the staged EMA immediately; the full RNG/recent/difficulty/history state is committed only after the optimizer update succeeds.
- Exact adaptive resume is covered by direct-continue versus serialize/load/continue equality. Batch-local curriculum changes are transactional: a failed collection/update restores a deep snapshot of RNG/difficulty/history, including an already-populated cache, while a successful update atomically renames the cache into place.
- Multi-group updates carry a stable `<run>:<update_index>` ID. The live trainer caches completed results, clears replayed stores, and does not execute the same logical update twice while that trainer process remains resident. A lost response reconnects and replays the same ID, returning the curriculum snapshot attached to the original optimized batch; an uncertain server-side failure poisons the ID instead of retrying it.
- Added a fail-closed static audit for the current piRL non-joint path: new and old log-prob terms must remain elementwise masked and `exp()` must operate on elementwise valid log-ratios.
- Added a canonical CPU-only dry run. It validates group size 8, at least 8 groups/logical-update, 2 logical update RPCs, 2 optimizer epochs/RPC, adaptive universe/cap compatibility, piRL non-joint source, and unique run naming. The same JSON now generates the exact trainer/client argv consumed by `multigroup-gpu-gate`; the dry-run is no longer disconnected from execution.

## Meaningful mutation coverage

The clean trace passes, while each mutation fails independently:

1. duplicate group seed
2. batch-global/changed group advantage
3. two logical update RPCs
4. omitted final group/store trajectories
5. trajectory-ID collision
6. joint-ratio reintroduction in trace
7. ratio/mask shape mismatch
8. new-logprob elementwise mask changed to a joint sum
9. old-logprob elementwise mask changed to a joint sum
10. ratio `exp(valid_lr)` changed to `exp(valid_lr.sum())`

The max-cap, stale-store, collection-exception reset, zero-advantage group, distinct seed, aggregate completeness, exact adaptive-resume, and partial-response EOF/replay paths also run as deterministic baseline tests.

## Commands run

    cd /home/aiden/Desktop/lab/robot/robocasa-docker/rl-train-t_3ed65912
    python tests/test_multigroup_batch.py
    python tests/test_multigroup_production_config.py
    python tests/test_update_batch.py
    python tests/test_training_correctness.py
    python tests/test_alllinear_smoke_gate.py
    python -m py_compile src/adaptive_curriculum.py src/update_batch.py src/grpo_train_loop.py src/grpo_trainer_server.py scripts/multigroup_production.py tests/test_multigroup_batch.py tests/test_multigroup_production_config.py
    bash -n scripts/run-train.sh

Observed result: 11 multi-group baseline tests passed; 4 production dry-run/source-mutation tests passed; existing update/training/smoke tests exited 0; syntax checks exited 0.

Canonical dry run:

    cd /home/aiden/Desktop/lab/robot/robocasa-docker/rl-train-t_3ed65912
    bash scripts/run-train.sh multigroup-dry-run <unique-run-name>

## Pending GPU gate

This audit does not establish production readiness. Task `t_eed2e2d9` must still perform a clean deploy and real v4 run with two logical update RPCs (two optimizer epochs each), then verify epoch-0 ratio, elementwise rollout/recompute match, nonfinite/dropped counts, adapter delta, post-step KL/clip/ESS, distinct group seeds, collected groups/trajectories/chunks, strict checkpoint mutation roundtrip, save/resume, and resource limits. The configured 1024 trainable-chunk floor may require more than 8 groups and is bounded by 24 groups; only the GPU artifact can determine whether that cap is feasible.

The update-ID replay cache is process-local. A trainer-process crash after an optimizer commit but before checkpoint/save remains a pending GPU interrupted-resume gate; this CPU task does not claim crash-durable exactly-once training or end-to-end model/optimizer resume.
