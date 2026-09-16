#!/usr/bin/env bash
# RLinf-MiBoT deployment run wrapper (raw `docker run`, mirrors xiaomi-cu121/run.sh style).
# An alternative to docker-compose for hosts without the compose plugin. Reads .env.
set -euo pipefail
cd "$(dirname "$0")"
root="$PWD"
repo_root="$(cd "$root/../.." && pwd)"

# --- load .env -------------------------------------------------------------
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
IMAGE="${RLINF_IMAGE:-rlinf-mibot:latest}"
GPUS="${RLINF_GPUS:-all}"
PORT="${RLINF_TRAINER_PORT:-10088}"
: "${RLINF_CHECKPOINT:?set RLINF_CHECKPOINT in .env}"
: "${RLINF_ASSETS:?set RLINF_ASSETS in .env}"
: "${RLINF_RESULTS:?set RLINF_RESULTS in .env}"
: "${RLINF_CACHE:?set RLINF_CACHE in .env}"
mkdir -p "$RLINF_RESULTS" "$RLINF_CACHE"

# Common mounts: integration layer + reused verified in-repo components (read-only),
# host checkpoint/assets read-only, results/cache read-write. GPU + EGL like the sim image.
common_args=(--gpus "$GPUS" --shm-size=2g --network none
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
  -e HF_HOME=/cache/hf -e HF_HUB_OFFLINE=1
  -v "$root:/integration:ro"
  -v "$repo_root/rl-env-t_4f3f2b20:/rl_env:ro"
  -v "$repo_root/rollouts-xiaomi-t_4a072806/tools:/skill_eval_tools:ro"
  -v "$RLINF_CHECKPOINT:/checkpoint:ro"
  -v "$RLINF_ASSETS:/opt/robocasa/robocasa/models/assets:ro"
  -v "$RLINF_RESULTS:/results"
  -v "$RLINF_CACHE:/cache")

case "${1:-help}" in
  build)
    docker build \
      --build-arg TORCH_CUDA_ARCH_LIST="${RLINF_CUDA_ARCH:-8.6}" \
      --build-arg MAX_JOBS="${RLINF_MAX_JOBS:-4}" \
      --build-arg RLINF_COMMIT="${RLINF_COMMIT:-bde6c918642abf9a4776cb1d5fabcc5087dfe195}" \
      -f docker/Dockerfile -t "$IMAGE" .
    ;;
  smoke)
    docker run --rm "${common_args[@]}" "$IMAGE" \
      python3 /integration/src/smoke.py --all --out /results/smoke "${@:2}"
    ;;
  ppo-smoke)
    docker run --rm "${common_args[@]}" "$IMAGE" \
      python3 /integration/src/train_entry.py --config-name ppo_smoke results_dir=/results/ppo_smoke "${@:2}"
    ;;
  ode-eval)
    docker run --rm "${common_args[@]}" "$IMAGE" \
      python3 /integration/src/eval_entry.py --config-name ode_eval results_dir=/results/ode_eval "${@:2}"
    ;;
  shell)
    docker run --rm -it "${common_args[@]}" --entrypoint bash "$IMAGE"
    ;;
  *)
    printf '%s\n' 'Usage: bash run.sh {build|smoke|ppo-smoke|ode-eval|shell}' >&2
    exit 2;;
esac
