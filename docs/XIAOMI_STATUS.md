# Expanded Xiaomi-Robotics-1 milestone: BLOCKED, NOT COMPLETE

The task's scope-change comment was discovered only during the final card read.
The earlier completion transition was premature for that expanded scope.
The simulator milestone is verified; Xiaomi deployment, actual-policy episode
and MP4 recording are NOT done. No zero-action smoke test is claimed to be a
model rollout, and no policy video has been generated.

## Official release verified

Project page: https://robotics.xiaomi.com/xiaomi-robotics-1.html
Official code: https://github.com/XiaomiRobotics/Xiaomi-Robotics-1
Inspected code commit: 0dd7aef8dc87296246aae812a1f59ccb708e5546
RoboCasa365 checkpoint:
https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365
API reports public, not gated, not disabled; revision:
3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4
The official collection has a distinct RoboCasa checkpoint too; do not confuse it
with RoboCasa365. No model-weight download or inference has yet been performed.

Official deployment guide: docs/DEPLOYMENT.md in the code repository.
Reference stack: Python 3.12, torch 2.8.0 / torchvision 0.23.0 / torchaudio 2.8.0
from cu128 wheels; transformers exactly 4.57.1; flash-attn 2.8.3; BF16.
The model card instead lists Python 3.11; resolve the build choice on resumption.
The model is advertised as 5B and consumer-GPU capable, but peak VRAM requirements
were not measured and no exact minimum VRAM claim is made here.

## Concrete runtime blocker on v4

Host NVIDIA driver is 535.183.01, reporting CUDA 12.2 capability.
The official CUDA-12.8 runtime preflight was actually executed:

    docker run --rm --gpus all nvidia/cuda:12.8.1-runtime-ubuntu22.04 nvidia-smi

The image downloaded, but container startup exited 125:

    nvidia-container-cli: requirement error: unsatisfied condition: cuda>=12.8,
    please update your driver to a newer version, or use an earlier cuda container

Full real output: logs/xiaomi-cuda-preflight.log.
No driver change, reboot, toolkit replacement or requirement-gate bypass was
attempted. This demonstrates a blocker for the documented CUDA-12.8 deployment
path; it does NOT prove every possible older-CUDA/non-reference stack is
impossible. Such an alternative is not validated here.

Required operator action: provide a CUDA-12.8-compatible NVIDIA driver/host and
reopen/resume the task, or explicitly choose a non-reference older-CUDA port for
separate compatibility testing. Shared-host driver changes require operator
coordination and may affect other workloads.

## Interface findings for resumption

Official evaluator: eval_robocasa365/entry.py.
- Three cameras: video.robot0_agentview_left, video.robot0_agentview_right,
  video.robot0_eye_in_hand.
- EE-first 14D state: relative EE position, relative orientation converted from
  XYZW quaternion to axis-angle, gripper qpos, base position and axis-angle.
  Padded to 60 state dimensions before processor input.
- Actions: first 12 dimensions returned by the official client; use checkpoint
  AutoProcessor and its robot-type-specific normalization/decode, never invent
  an action scale or transplant statistics from the RoboCasa-only checkpoint.
- Reference observation history 4, interval 2, 16 actions per query, crop 0.95,
  pretrain split, base seed 7. Gym reset must accept a seed.
- Existing simulator smoke validates PandaOmron but not the full XR-1 Gym
  observation schema. Check state.* versus body.*/hand.* keys before integration;
  do not infer compatibility merely from the same environment version number.

Proposed task: CloseBlenderLid, seed 7, one episode, because the official README
uses this task for its single-task smoke example. This is a candidate, not a
completed rollout. Record the actual instruction from
observation['annotation.human.task_description'] at reset rather than inventing
it; retain success/failure, step count, horizon/termination and a decodable MP4.
Use a full task horizon for the requested episode, not just the README's 20-step
connection smoke test.

Continue in separate inference/client Docker layers based on the existing
verified simulation image. Avoid scripts/deploy.sh as-is on the host: it kills
the generic model_servers tmux session; launch a task-scoped container directly
with deploy/server.py instead. Do not expose its socket to untrusted networks.

## Preserved work

Local: /home/aiden/Desktop/lab/robot/robocasa-docker
Remote: /home/v4/robocasa-docker-t_04bf4fb7
Simulator image: robocasa-sim:t_04bf4fb7
Assets: robocasa-assets-t_04bf4fb7 (22G)
Simulation build, GPU/CPU rendering and reset/step evidence remain valid.
README.md and VERIFICATION.md document that milestone only.
