# Z-1 expert adapter smoke test — pre-fix diagnostic

Experiment ID: `EXP-20260917-Z1-EXPERT-SMOKE`
Run: `alllinear_grasp_smoke2`
Classification: **PRE-FIX CAPACITY / THROUGHPUT DIAGNOSTIC ONLY**
Performance claim: **none** (`--skip-eval`; zero optimizer updates completed)

## Configuration

- Base/model: Xiaomi-Robotics-1-RoboCasa365
- Task/skill: CloseBlenderLid / GRASP
- Adapter: action-expert all-linear LoRA, rank 8, alpha 32
- Loaded targets/trainables: 189 modules / 6.35M parameters
- Sampler: piRL Flow-SDE, 5 denoise steps, eta 0.1
- Horizon/control/chunk: 208 steps / 20 Hz / 16 actions
- Group: 8 trajectories
- Batch target: minimum 8 groups and 1024 nonzero-advantage chunks; maximum 24 groups
- Requested updates: 2
- Optimizer: AdamW, lr 1e-5, weight decay 0.01, grad clip 1.0
- Update objective: one on-policy group-relative policy-gradient epoch (`update_epochs=1`); PPO clipping was inactive and KL coefficient was 0
- Reward in this pre-fix run: terminal success plus nominal 20-step hold; audit found missing completion decay and a hold-lapse termination bug

## Observed result

- Completed optimizer updates: **0 / 2**
- `train_log.jsonl`: 0 rows
- Checkpoint: not created
- `run_summary.json`: not created
- Final eval: not run (`--skip-eval`)
- Final success/reward/PPO/adapter-delta metrics: unavailable because no update completed
- Live operator observation during collection: 36 rollouts started, mean start interval 22.17 s, provisional maximum 468 / 1024 trainable chunks
- Captured `train.log`: 22 environment starts before the original SSH/tee stream detached; it is not a complete rollout count
- Observed GPU allocation: up to 10,490 MiB (point observation, not an instrumented peak)
- OOM: not observed while the trainer remained resident
- NaN status: not measurable because optimizer/recompute never ran
- End state: client exited before update 1; trainer store was verified empty afterward; trainer was then stopped and GPU memory released

## Gate

**FAIL — capacity threshold not reached.** The requested 1024 trainable chunks did not produce even one completed update within the 24-group cap. This run does not validate learning, PPO clipping, reward correctness, checkpoint safety, or policy improvement.

Do not start a 30-update run from this code. Wait for `t_2bb64447`, which fixes reward decomposition/decay, exact-N hold termination, mean chunk loss, raw ratio guarding, progress logging, strict named checkpoints, and source-hash gating, then run a small exact-source GPU micro-smoke before choosing a feasible batch size.

## Preserved remote artifacts

- `/home/v4/rl-train-t_3ed65912/results/alllinear_grasp_smoke2/train.log`
- `/home/v4/rl-train-t_3ed65912/results/alllinear_grasp_smoke2/train_log.jsonl` (empty)
- `/home/v4/rl-train-t_3ed65912/results/alllinear_grasp_smoke2/trainer_server.log`

## Source provenance warning

The pre-fix deployment had no Git metadata/source manifest, and its hashes differ from the current canonical tree. Therefore the run is intentionally preserved as a non-reproducible pre-fix diagnostic.

| File | pre-fix v4 SHA-256 | current canonical SHA-256 at report time |
|---|---|---|
| `src/lora.py` | `5f63742e8130c2eff58a7fc0a28bca10752988c7efb42794f054b8414c1be510` | `410c495ec38fa44d8966a0c4c4125b3c7a10f92968c385e128bb10c1a711c3b9` |
| `src/grpo_train_loop.py` | `527bd54afd618c5adae1d0a17a62a80fd631d697f431daca6f0f7a21e451cf6f` | `2288466bfebfeb01de83cb01cafff6b1b9599d57be4d5670ae7eb31a5e775112` |
| `src/grpo_trainer_server.py` | `ee6a7297dbb58e05e4b1a2b9ed1234cef0ca9d7ffbfc4be90aeacca8c31e1d7e` | `d331c4ec9601f3dbf867a9c895cf996b8820acabb837b20b1a35f4040da90446` |
| `src/update_batch.py` | `3dd7674c666e0a10a4bd25c6a3e06ba92d27de1e4bc437745068a32605666395` | `3702761144d49f6f3cae75219a402343f233afd3df53540ced90698800d37852` |
