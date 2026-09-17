# GRASP action-expert all-linear LoRA — corrected micro gate

Task: `t_4c421da6`
Final code: `6fc53e6` (checkpoint-path fix base `3154248` + RNG restore fix)
Classification: correctness/capacity validation, not a performance comparison

## Pre-fix requested smoke

- Run: `alllinear_grasp_smoke2`
- Result: 0/2 optimizer updates; provisional live maximum 468/1024 trainable chunks.
- No checkpoint, summary, paired evaluation, or learning claim.
- Decision: gate FAIL; the pre-fix 30-update continuation was prohibited.
- W&B: the current mirror does not include this run.

## Exact-source corrected micro

Artifact: `/home/v4/rl-train-t_3ed65912/results/corrected_exact_3154248_micro_lr1e6/`

- Source: clean commit `3154248851cceb3703b9b60931ce1e1627282bbd` in both train and env manifests.
- Model/adapter: Xiaomi-Robotics-1-RoboCasa365; GRASP; all-linear action-expert LoRA rank 8, alpha 32.
- Targets/trainables: 189 modules; 6,352,288 / 5,059,501,984 parameters (0.1256%). VLM was not adapted.
- Sampler/update: piRL Flow-SDE, eta 0.1, group 8, 2 groups, 16 trajectories, 82 trainable chunks, AdamW 1e-6, two requested update epochs, target KL 0.05.
- Reward signal: one all-success group was gated; one mixed group had 7/8 successes. Total successes/hold successes: 15/15.
- Timing: collection 424.1 s; optimizer 271.0 s; 26.51 s/trajectory; 0.3715 s/stored chunk.
- Peak allocated GPU memory: 10.52 GiB. OOM: no. Nonfinite/dropped chunks: 0/0.
- Correctness: epoch-0 recompute ratio 1.0; adapter delta L2 0.001957 > 0; run-unique checkpoint created in the run directory.
- Update stability: epoch-1/post-step ratio 0.8972, KL 0.2903, clip fraction 0.7927, ESS 0.4849. Target-KL early-stop fired only after this overshoot.

## Checkpoint roundtrip

The first real load exposed a bug: `torch.load(..., map_location=cuda)` moved the saved CPU RNG ByteTensor to CUDA, and `torch.set_rng_state` rejected it. Commit `6fc53e6` normalizes CPU and CUDA RNG state tensors to CPU before restore. Verification:

- CPU unit test: checkpoint schema and RNG normalization PASS.
- Mutation test: removing CPU normalization is caught.
- Central/v4 SHA-256 match for `checkpoint_schema.py` and `grpo_trainer_server.py`.
- Real v4 strict RPC load of the corrected run checkpoint PASS: schema 2, 378 named LoRA tensors, 0 extra tensors, no missing/unexpected/layout error.

## Final gate and decision

The RTX 3090 forward/backward/update capacity gate passes for the 6.35M-parameter all-linear adapter, but the production-learning gate fails. Even at AdamW 1e-6 the first off-policy epoch overshot target KL badly (0.2903) with 79.3% clipping. Therefore no 30-update run was started. The original qkv-only paired/held-out comparison is unavailable because proceeding would produce an unsafe and scientifically invalid performance claim. The next optimizer design must prevent the first changed epoch from overshooting (rather than merely stopping afterward), then rerun paired/held-out evaluation.
