# amp_csi RTX3090 single-node GPU portability smoke

## Verdict

PASS for the bounded x86_64 / RTX3090 software-runtime path. This is not DGX Spark, ARM64, Blackwell, unified-memory, multi-node, RoCE, or topology validation; those gates remain open on task `t_574dfcc7`.

The tested implementation began at source commit `d67207d`. Runtime defects found during the smoke were fixed on branch `wt/t_43b4edc9`. The measured successful rollout/update process used source commit `e79e7f2272c4325cae23139a84ec5c8f85641bc4`; comparison/postflight evidence was collected at `471dd203639bd6b125fc5548423c54e5329ecaed`. Final QA fixes and preserved evidence end at the commit containing this report.

## Frozen measured contract

- Host: `amp2`, x86_64, NVIDIA GeForce RTX 3090 24 GiB, driver 580.95.05.
- Image: `rlinf-mibot:t_43b4edc9`, image ID `sha256:e8c0b248afb6fe759616049cfaa45c147d6ac56cfdf79788063577d735dc073c`.
- Base checkpoint: `/home/guest/robocasa-docker-t_9f03a613/checkpoint`; all 20 files passed the committed SHA-256 manifest.
- Assets: task-scoped writable overlay volume over the validated `robocasa-assets-t_5af7225b` parent; parent was never made writable and the task volume was removed postflight.
- Grid: two noise configs (`0.1`, `0.3`), disjoint seed bands, one real episode per config, effective horizon 250 and replan interval 16.
- Trainer: AdamW, LR `1e-5`, adapter-only rank 8 / alpha 32, update clip `0.1`, KL coefficient `0.005`, ratio max `10`, advantage clip `3`, target KL `0.02`.
- Consume gate: one accepted update RPC over 28 real chunks, two configured update epochs, then checkpoint save and strict load.

The launcher commands attempted `env.horizon=16 rollout.replan_steps=8`, but the original dotted-key override parser did not apply those values to the canonical dotted grid keys. The artifacts correctly show the effective values (250/16). Commit `c91ac71` fixes and tests this parser for future runs; no measured result is relabeled with the intended override.

## Measured results

### Serial baseline

- Run: `t43-serial-v2`
- Status: done; audit PASS.
- Configs / episodes: 2 / 2.
- Max concurrency: 1.
- Elapsed: 67.593 s; 106.519 episodes/hour.
- Peak trainer allocation: 10.40 GiB.
- Duplicate seeds, config mixing, job mismatches, collector optimizer requests: all 0.

### Two-worker parallel collection

- Run: `t43-parallel-pass`
- Status: done; audit PASS.
- Distinct RLinf workers observed: 2.
- Max concurrency: 2.
- Cross-config overlap: 37.783069372177124 s.
- Elapsed: 39.558 s; 182.009 episodes/hour.
- Measured speedup over the paired serial run: 1.709x.
- Peak trainer allocation: 10.41 GiB.
- Duplicate seeds, config mixing, job mismatches, collector optimizer requests: all 0.
- Payload hashes and rollout artifacts: valid.

### Failure, retry, resume, and duplicate launch

- Planned seed 710000 failure was recorded once in `injected-failures.jsonl` and succeeded on the single bounded retry (`retry_count=1`, two completed episodes). Because the first worker returned immediately, that failure-injection run intentionally did not satisfy overlap; overlap was established separately by `t43-parallel-pass`.
- `t43-resume-test` was interrupted after one durable episode (collector exit 137), then resumed after live-container verification. The final audit has exactly two unique jobs and no duplicate seed/job IDs, proving the completed job was skipped.
- A second `model-start t43-parallel` was refused with exit 1 while the original model container was live.

### Trainer consume / update gate

- Collector phase requested 0 optimizer updates.
- Imported payloads: 2 trajectories, 28 chunks.
- Accepted update RPCs: 1; replayed update: false.
- Epoch-0 mean ratio: 1.0.
- Adapter delta L2: 0.01762840710580349.
- Nonfinite / dropped counts: 0 / 0.
- Final mean KL: 0.00014759279743191742.
- Final clip fraction: 0.006138392857142857.
- Final ESS: 0.9997067189194196.
- Post-step peak memory: 10.47 GiB.
- Checkpoint: 216 LoRA tensors, optimizer and RNG state present; save and strict load both passed.

## Defects removed during the real smoke

1. Added task-scoped Docker asset-volume support and explicit `ro`/`rw` mode while keeping `ro` as the default.
2. Added missing RoboCasa runtime dependencies and corrected the strict NumPy pin to 2.2.5.
3. Made source-manifest generation independent of the caller's working directory.
4. Fixed mixed launcher-option / Hydra-override CLI parsing.
5. Fixed the canonical simulator snapshot call signature.
6. Converted RLinf actor failures to serializable envelopes so bounded retry works instead of killing the collector process.
7. Ran result comparison and stale-lock cleanup through the mounted container to handle root-owned result artifacts.
8. Fixed dotted grid-key override routing and added regression coverage.

## Postflight

- Experiment model/collector containers: none.
- Experiment compute processes: none.
- Ports 10086-10090: no listeners.
- GPU: 38 MiB used, 24085 MiB free, 0% utilization at recorded postflight.
- Protected unrelated container `gpu-nccl`: still running with unchanged container ID `1d279e38a9a70b434933e018e18f1a8aaa7c6472fde9037f103537149eb35eac`.

Raw JSON and compact logs in this directory are copied from `/home/guest/experiments/rlinf-t_43b4edc9` on `amp2`.