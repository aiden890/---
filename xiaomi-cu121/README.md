# Xiaomi-Robotics-1 / CUDA 12.1 experiment (t_9f03a613)

Status: COMPLETE for CUDA 12.1 port, actual policy inference and one full episode.
Robot task outcome: FAILURE (CloseBlenderLid, seed 7, 900 steps, horizon reached).
The policy MP4 was fully decoded: 451 frames, 768x256, 20 fps. See REPORT.md and
output/verification.json. A failed robot task is not being represented as success.

Execution host: v4. Local source/evidence is this directory. Remote working copy:
/home/v4/robocasa-docker-t_9f03a613

Pinned upstream code: https://github.com/XiaomiRobotics/Xiaomi-Robotics-1
commit 0dd7aef8dc87296246aae812a1f59ccb708e5546.
The unmodified evaluator and deploy client/server are under upstream/.
rollout.py differs only in its local import root and additional episode metadata
(instruction, terminal flags, termination reason). Policy preprocessing, action
normalization, action horizon and environment conversion remain upstream code.

Pinned checkpoint: XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365
revision 3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4 (public, not gated).
Remote checkpoint/ holds downloaded weights. Local checkpoint-api.json records
the revision. No substitute model or fabricated policy output is used.

Port configuration:
- NVIDIA CUDA 12.1.1 devel Ubuntu 22.04, base digest pinned in Dockerfile.
- Python 3.10 (Ubuntu package), torch 2.5.1+cu121, torchvision 0.20.1+cu121,
  torchaudio 2.5.1+cu121; transformers 4.57.1; flash-attn 2.8.3 from source.
- This is NOT the official Python 3.12 / torch 2.8.0 / CUDA 12.8 combination.
- Flash Attention source build uses nvcc 12.1, MAX_JOBS=4. The upstream extension
  chooses its own compile targets; TORCH_CUDA_ARCH_LIST is not a guarantee that
  only sm86 is built. Compiling backward kernels too can take considerable time.
- Client image layers CPU torch and transformers on the verified RoboCasa image.
- Build-time package freeze is inside each image; base OS packages and transitives
  are not all pre-resolved, so retain tested image IDs and resolved freeze files.

Safety/isolation:
- No host driver update, reboot, CUDA requirement-gate bypass or other server.
- robocasa-sim:t_04bf4fb7 is not retagged or modified.
- Original robocasa-assets-t_04bf4fb7 is read-only when copied. This task uses
  robocasa-assets-t_9f03a613 with a separate official generative-texture download.
- The original asset volume lacked generative textures. Mounting a subdirectory
  under that read-only volume failed because the mountpoint did not exist. A
  cloned volume avoids modifying the verified source volume.
- The model socket uses pickle. Server has --network none and binds loopback;
  the client joins that network namespace. No host port is exposed.
- run.sh stop stops ONLY the task-scoped model server, not generic tmux sessions.

Run on v4 from the remote working directory (not from the local Docker daemon):

    bash run.sh build
    bash run.sh download
    bash run.sh textures
    bash run.sh clone-assets
    bash run.sh observe
    bash run.sh probe
    bash run.sh server
    docker logs xiaomi-server-t_9f03a613
    # Wait for 'Model loaded' / server readiness before starting client.
    bash run.sh rollout
    bash run.sh verify
    bash run.sh stop

The server container is kept after stopping for evidence. A second server launch
needs a new task-scoped name or explicit removal of that stopped container.
Never start duplicate servers on the same GPU unintentionally.

Preflight uses a real CloseBlenderLid/pretrain reset, seed 7, three 256x256
cameras, official EE-first 14D state padded to 60, observation history 4 and crop
0.95. The natural language instruction is read from reset. The full task horizon
is 900 steps; no 20-step smoke horizon is used for the policy episode.

The upstream evaluator derives episode_seed = base seed + task index * trials +
episode index. Thus the rollout seed can differ from preflight seed 7; use the
actual episode record, not an assumed base seed. Success is the environment's
info.success; failure at the full horizon remains a completed but unsuccessful
policy episode, not a software-success or benchmark-success claim.

Checks and evidence:
- logs/cuda-preflight.log: container starts with unchanged 535.183.01 driver.
- logs/torch-cuda-check.log: real CUDA matrix multiplication and BF16 capability.
- logs/processor-check.log: checkpoint processor accepts real simulator observation.
- logs/observe*.log: original missing-texture/mount failures and successful reset.
- logs/build.log and client-build.log: actual build output, not upstream examples.
- test_outcome.py: terminal-reason classifier; red and green logs retained.
- output/inference.json and output/rollout/: exist only after corresponding tests.

Some Gym observation bounds are narrower than real world-space state values;
the upstream checker emits a warning on reset. This experiment asserts presence,
shape and finite values, and does not suppress or rewrite observation values.
No benchmark success rate can be inferred from a single episode.
