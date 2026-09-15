#!/usr/bin/env bash
# t_8cd8d8c0: instruction-attendance probe for Xiaomi-Robotics-1 RoboCasa365.
# Same server/image/checkpoint/assets as the verified t_4a072806 skill-eval run.
# Reuses /work/rollout.py and /skilltools/skill_eval.py by import (no forked policy code).
set -euo pipefail
parent=/home/v4/robocasa-docker-t_9f03a613
root=/home/v4/experiments/instr_attend_t_8cd8d8c0
skilltools=/home/v4/rollouts-xiaomi-t_4a072806/tools
server=xiaomi-server-t_460aea68
assets_volume=robocasa-assets-t_5af7225b
run=${1:-run1}
shift || true
out="$root/$run"
mkdir -p "$out"
docker ps --format "{{.Names}}" | grep -q "^$server$" || { echo "server $server not running" >&2; exit 3; }
docker ps --format "{{.Names}}" | grep -q "^xiaomi-client-" && { echo "a xiaomi-client is already running" >&2; exit 4; }
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader > "$out/gpu_before.txt"
entry=${ENTRY:-instr_probe.py}
docker run --rm --name "xiaomi-client-t_8cd8d8c0-$run" --gpus all --shm-size=2g \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v "$assets_volume:/opt/robocasa/robocasa/models/assets" -v "$parent:/work:ro" \
  -v "$skilltools:/skilltools:ro" -v "$root/tools:/tools:ro" -v "$out:/output" \
  -v "$parent/checkpoint:/checkpoint:ro" \
  --network "container:$server" xiaomi-client:t_9f03a613 \
  python /tools/"$entry" --out /output --seed 9 "$@" 2>&1 | tee "$out/${entry%.py}.log"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader > "$out/gpu_after.txt"
