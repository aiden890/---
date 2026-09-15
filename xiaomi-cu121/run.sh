#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=$PWD
image=xiaomi-cu121:t_9f03a613
client=xiaomi-client:t_9f03a613
server=xiaomi-server-t_9f03a613
sim_args=(--gpus all --shm-size=2g -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
  -v robocasa-assets-t_9f03a613:/opt/robocasa/robocasa/models/assets:ro
  -v "$root:/work:ro" -v "$root/output:/output" -v "$root/checkpoint:/checkpoint:ro")
mkdir -p logs output
case "${1:-help}" in
  build)
    docker build -t "$image" . 2>&1 | tee logs/build-final.log
    docker build -f Dockerfile.client -t "$client" . 2>&1 | tee logs/client-build-final.log
    ;;
  download)
    mkdir -p checkpoint
    docker run --rm -v "$root/checkpoint:/checkpoint" "$image" python3 /app/download.py 2>&1 | tee logs/download.log
    ;;
  textures)
    docker run --rm -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa -v "$root:/work" robocasa-sim:t_04bf4fb7 python /work/textures.py 2>&1 | tee logs/textures.log
    ;;
  clone-assets)
    docker volume create robocasa-assets-t_9f03a613
    docker run --rm -v robocasa-assets-t_04bf4fb7:/source:ro -v robocasa-assets-t_9f03a613:/destination -v "$root/extra-assets:/extra:ro" python:3.11.14-slim-bookworm sh -c 'cp -a /source/. /destination/ && cp -a /extra/generative_textures /destination/' 2>&1 | tee logs/assets-copy.log
    ;;
  observe)
    docker run --rm "${sim_args[@]}" robocasa-sim:t_04bf4fb7 python /work/observe.py 2>&1 | tee logs/observe-final.log
    ;;
  probe)
    docker run --rm --gpus all --network none -v "$root/checkpoint:/checkpoint:ro" -v "$root:/work:ro" -v "$root/output:/output" "$image" python3 /work/probe.py 2>&1 | tee logs/probe.log
    ;;
  server)
    docker run -d --name "$server" --gpus all --network none --shm-size=2g -v "$root/checkpoint:/checkpoint:ro" -v "$root:/work:ro" "$image" python3 /work/upstream/deploy/server.py --model /checkpoint --host 127.0.0.1 --port 10086
    ;;
  rollout)
    docker run --rm "${sim_args[@]}" --network "container:$server" "$client" python /work/rollout.py --model-path /checkpoint --server-addr 127.0.0.1 --server-port 10086 --task-name CloseBlenderLid --num-trials 1 --seed 7 --save-videos --save-root-dir /output --run-id rollout 2>&1 | tee logs/rollout.log
    ;;
  verify)
    docker run --name xiaomi-verify-t_9f03a613 -v "$root:/work:ro" -v "$root/output:/output" "$client" python /work/verify.py 2>&1 | tee logs/verification.log
    ;;
  stop)
    docker logs "$server" > logs/server.log 2>&1
    docker stop "$server"
    ;;
  *) printf '%s\n' 'Usage: bash run.sh {build|download|textures|clone-assets|observe|probe|server|rollout|verify|stop}' >&2; exit 2;;
esac
