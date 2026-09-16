#!/usr/bin/env bash
# t_1d5b404f: adapter-free full-architecture inference smoke test (CloseBlenderLid).
# Reuses the SAME server / images / checkpoint / assets as t_4a072806.  The
# architecture_smoke/ package imports the unchanged /work/rollout.py and the
# parent skill_eval.Sim (predicate single source of truth) inside the container.
set -euo pipefail
parent=/home/v4/robocasa-docker-t_9f03a613
root=/home/v4/architecture-smoke-t_1d5b404f
server=xiaomi-server-t_460aea68
assets_volume=robocasa-assets-t_5af7225b
seeds=${1:-0,1,2}
out="$root/out"
mkdir -p "$out"
docker ps --format "{{.Names}}" | grep -q "^$server$" || { echo "server missing" >&2; exit 3; }
docker ps --format "{{.Names}}" | grep -q "^xiaomi-client-" && { echo "client already running" >&2; exit 4; }
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$out/gpu_before.txt"
# skill_eval.py must be importable as a module: mount the parent rollouts tools too.
docker run --rm --name "xiaomi-client-t_1d5b404f" --gpus all --shm-size=2g \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e PYTHONPATH=/pkg:/tools:/work \
  -v "$assets_volume:/opt/robocasa/robocasa/models/assets" -v "$parent:/work:ro" \
  -v "$root/architecture_smoke:/pkg:ro" \
  -v "/home/v4/rollouts-xiaomi-t_4a072806/tools:/tools:ro" \
  -v "$out:/output" -v "$parent/checkpoint:/checkpoint:ro" \
  --network "container:$server" xiaomi-client:t_9f03a613 \
  python /pkg/run_architecture_smoke.py --out /output --seeds "$seeds" \
    --model-path /checkpoint --server-addr 127.0.0.1 --server-port 10086 \
    2>&1 | tee "$out/run.log"
