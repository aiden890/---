# Verified result: t_9f03a613

## Outcome

The CUDA 12.1 port, checkpoint load, real policy inference, and one full RoboCasa365
policy episode are COMPLETE. The robot task itself FAILED. There is no claim of
successful lid closure or benchmark performance.

- Task: CloseBlenderLid; split: pretrain; episode seed: 7.
- Instruction from reset: `Close the lid blender by securely placing the lid on top.`
- Actual policy: XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365.
- Episode: 900 steps; success=false; termination_reason=horizon;
  done=false; truncated=false.
- Video: output/rollout/CloseBlenderLid/episode_000_seed_7_failure.mp4
- Full-file decoding: 451 frames, 768x256 RGB, 20 fps; 1,309,724 bytes.
- Video SHA256: a1d929eeb1141569d490ec5242ac8e7dff9d58be5f35f75e0fc74d7745f98909
- Video stride 2 matches upstream defaults. This is not a real-time-speed claim.
- First and last frames were visually inspected: real nonblank three-camera
  kitchen views, with robot/camera pose changes. Visual inspection does not
  override the environment's failure result.

## Runtime and inference evidence

Host: v4, RTX 3090 24 GB, unchanged NVIDIA driver 535.183.01.
Container: CUDA toolkit 12.1.1, Python 3.10.12, PyTorch 2.5.1+cu121,
transformers 4.57.1, flash-attn 2.8.3 compiled from source with nvcc 12.1.

Inference image:
sha256:aac5e8e597636dc80df55a3150da68f2fb5897e28bfb02836319578fba477c87

- Basic CUDA matrix multiplication passed and BF16 support was detected.
- A real flash_attn_func CUDA kernel executed and returned finite values.
- Loaded 5,053,149,696 checkpoint parameters; load time 2.7972 seconds.
- Real reset observation was processed using upstream messages, state ordering,
  processor normalization and decoding. No mock observations or random policy.
- Single model inference returned finite 16x12 decoded actions in 1.2636 seconds.
- Torch peak allocated memory for this preflight: 10,398,234,624 bytes (~9.7 GiB).
  This is allocator memory, not total device memory or an all-episode peak.
- During-rollout nvidia-smi sample: 12,136 MiB total device memory, GPU utilization
  48%; model process 10,448 MiB and simulator graphics process 1,656 MiB.
  These are point samples, not peak estimates.

The server ran the unmodified upstream deploy/server.py on loopback in a network
namespace with no external network and no published ports. The client used the
same namespace and official client protocol. The evaluation loop is copied from
upstream with only local import-root and result-metadata additions.

## Pinned inputs and reproducibility

Code commit: 0dd7aef8dc87296246aae812a1f59ccb708e5546
Checkpoint revision: 3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4
checkpoint.sha256 records SHA256 values for downloaded checkpoint files.

Dockerfile and Dockerfile.client, exact resolved package manifests, scripts,
unmodified upstream client/server/evaluator, logs and artifacts are preserved
here. The actual 9.5G weights are on v4 in checkpoint/; download.py obtains the
same fixed revision. README.md contains build and rerun commands.

This is a validated non-reference stack, not Xiaomi's official torch2.8/cu128
stack. Full transitives are in inference.requirements.lock; notably PyTorch's
resolver selected nvidia-nvjitlink-cu12 12.9.86 while torch's CUDA runtime and
compiler are 12.1. The actual GPU kernel and policy checks above passed with this
resolved combination. The manifest records that fact rather than claiming every
NVIDIA package version equals 12.1. Builds still depend on mutable apt repositories;
keep the tested image for a byte-identical environment.

## Tests, failures and limitations

Passed: CUDA container preflight, CUDA matmul, model class import, real observation
schema, official processor input, source-built Flash Attention kernel, checkpoint
loading, finite decoded policy actions, 900-step actual rollout, complete MP4
frame decoding, local/remote MP4 checksum equality, terminal-reason unit test,
original simulator image identity, and stopped model-server state.

The acceptance test was executed before inference and correctly failed on missing
output/inference.json. It later passed on the actual generated episode/video.

Inference image `pip check`: no broken requirements. Client `pip check`: lerobot
and tianshou are omitted, inherited from the deliberate simulator-only base;
these optional training/conversion dependencies were not needed for the verified
rollout. No training or dataset conversion support is claimed.

Preserved failure logs:
- observe.log: missing generative_textures/wall/tex071.png at reset.
- observe-with-textures.log: nested bind mount under a read-only volume failed
  because the target directory did not exist.
- Resolved by downloading official generative textures and cloning original
  assets to robocasa-assets-t_9f03a613, not by disabling textures or faking assets.

Nonfatal upstream warnings remain visible: narrow Gym observation bounds,
missing optional robot modules, unspecified video metadata defaulting to 24 fps
inside processor sampling, and a Flash Attention dtype warning despite explicit
BF16 casting. The decoder validation also emits a NumPy deprecation warning.
None was silently hidden; execution outcomes above were directly checked.

The one episode failed at its full horizon. Diagnosing policy/task success,
tuning hyperparameters, or evaluating more seeds is outside this card.

## Isolation after completion

Model server xiaomi-server-t_9f03a613 is stopped (State.Status=exited,
State.Running=false). Its logs remain. Task-specific images, stopped diagnostic
containers, checkpoint and asset volume remain for reproduction.

Original simulator image is unchanged:
sha256:83c1bc264626107a232003c874cbf7725de4988ddf1e417ac0abfb6aa3a52aba

No host driver update, reboot, requirement-gate bypass, other server, alternative
policy, unrelated tmux session or existing verified artifact modification.
