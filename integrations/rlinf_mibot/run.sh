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
SOURCE_COMMIT="$(git -C "$repo_root" rev-parse HEAD)"
: "${RLINF_CHECKPOINT:?set RLINF_CHECKPOINT in .env}"
: "${RLINF_ASSETS:?set RLINF_ASSETS in .env}"
: "${RLINF_RESULTS:?set RLINF_RESULTS in .env}"
: "${RLINF_CACHE:?set RLINF_CACHE in .env}"
mkdir -p "$RLINF_RESULTS" "$RLINF_CACHE"
ASSETS_MODE="${RLINF_ASSETS_MODE:-ro}"
[[ "$ASSETS_MODE" == ro || "$ASSETS_MODE" == rw ]] || {
  echo "RLINF_ASSETS_MODE must be ro or rw" >&2; exit 2
}
assets_mount="$RLINF_ASSETS:/opt/robocasa/robocasa/models/assets:$ASSETS_MODE"

validate_run_id() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || {
    echo "invalid RUN_ID (allowed: A-Za-z0-9._-, max 128, no path separators): $1" >&2
    exit 2
  }
}

# Common mounts: integration layer + reused verified in-repo components (read-only),
# host checkpoint/assets read-only, results/cache read-write. GPU + EGL like the sim image.
common_args=(--gpus "$GPUS" --shm-size=8g
  -e RLINF_SOURCE_COMMIT="$SOURCE_COMMIT"
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
  -e HF_HOME=/cache/hf -e HF_HUB_OFFLINE=1
  -v "$root:/integration:ro"
  -v "$repo_root/rl-env-t_4f3f2b20:/rl_env:ro"
  -v "$repo_root/rl-train-t_3ed65912:/train:ro"
  -v "$repo_root/xiaomi-cu121:/work:ro"
  -v "$repo_root/rollouts-xiaomi-t_4a072806/tools:/skill_eval_tools:ro"
  -v "$RLINF_CHECKPOINT:/checkpoint:ro"
  -v "$assets_mount"
  -v "$RLINF_RESULTS:/results"
  -v "$RLINF_CACHE:/cache")

case "${1:-help}" in
  build)
    docker build \
      --build-arg CUDA_BASE_IMAGE="${RLINF_BASE_IMAGE:-nvidia/cuda:12.1.1-devel-ubuntu22.04@sha256:7012e535a47883527d402da998384c30b936140c05e2537158c80b8143ee7425}" \
      --build-arg INSTALL_TORCH="${RLINF_INSTALL_TORCH:-1}" \
      --build-arg INSTALL_FLASH_ATTN="${RLINF_INSTALL_FLASH_ATTN:-1}" \
      --build-arg TORCH_CUDA_ARCH_LIST="${RLINF_CUDA_ARCH:-8.6}" \
      --build-arg MAX_JOBS="${RLINF_MAX_JOBS:-4}" \
      --build-arg RLINF_COMMIT="${RLINF_COMMIT:-bde6c918642abf9a4776cb1d5fabcc5087dfe195}" \
      -f docker/Dockerfile -t "$IMAGE" .
    ;;
  smoke)
    docker run --rm --network none "${common_args[@]}" "$IMAGE" \
      python3 /integration/src/smoke.py --all --out /results/smoke "${@:2}"
    ;;
  ppo-smoke)
    docker run --rm --network none "${common_args[@]}" "$IMAGE" \
      python3 /integration/src/train_entry.py --config-name ppo_smoke results_dir=/results/ppo_smoke "${@:2}"
    ;;
  ode-eval)
    docker run --rm --network none "${common_args[@]}" "$IMAGE" \
      python3 /integration/src/eval_entry.py --config-name ode_eval results_dir=/results/ode_eval "${@:2}"
    ;;
  shell)
    docker run --rm -it --network host "${common_args[@]}" --entrypoint bash "$IMAGE"
    ;;
  model-start)
    run_id="${2:?Usage: bash run.sh model-start RUN_ID}"
    validate_run_id "$run_id"
    name="rlinf-mibot-model-${run_id}"
    if docker ps --format '{{.Names}}' | grep -qx "$name"; then
      echo "refusing duplicate launch: $name is already running" >&2; exit 1
    fi
    if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
      echo "refusing stale container reuse: remove $name after preserving its logs" >&2; exit 1
    fi
    mkdir -p "$RLINF_RESULTS/$run_id/provenance"
    python3 "$repo_root/scripts/build_source_manifest.py" \
      rl-train-t_3ed65912 "$RLINF_RESULTS/$run_id/provenance/train.json"
    python3 "$repo_root/scripts/build_source_manifest.py" \
      rl-env-t_4f3f2b20 "$RLINF_RESULTS/$run_id/provenance/env.json"
    docker run -d --name "$name" --network host "${common_args[@]}" "$IMAGE" \
      python3 /train/src/grpo_trainer_server.py --model /checkpoint --host 127.0.0.1 --port "$PORT" \
      --train-source-manifest "/results/$run_id/provenance/train.json" \
      --env-source-manifest "/results/$run_id/provenance/env.json" \
      --sampler pirl --eta 0.1 --optimizer adamw --lr 1e-5 --update-epochs 1
    for _ in $(seq 1 180); do
      if docker logs "$name" 2>&1 | grep -q 'GRPO trainer server on'; then
        echo "model server ready: $name"; exit 0
      fi
      if [[ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null || true)" != true ]]; then
        docker logs "$name" >&2; exit 1
      fi
      sleep 5
    done
    echo "model server readiness timeout" >&2; docker logs "$name" >&2; exit 1
    ;;
  model-stop)
    run_id="${2:?Usage: bash run.sh model-stop RUN_ID}"
    validate_run_id "$run_id"
    docker stop "rlinf-mibot-model-${run_id}" >/dev/null
    docker rm "rlinf-mibot-model-${run_id}" >/dev/null
    ;;
  grid-parallel|grid-serial|grid-resume)
    cmd="$1"; run_id="${2:?Usage: bash run.sh $1 RUN_ID [OmegaConf overrides...]}"; shift 2
    validate_run_id "$run_id"
    name="rlinf-mibot-collector-${run_id}"
    [[ ! -e "$RLINF_RESULTS/$run_id/DONE" ]] || { echo "DONE exists; refusing overwrite" >&2; exit 1; }
    if [[ "$cmd" == grid-resume ]]; then
      if docker ps --format '{{.Names}}' | grep -qx "$name"; then
        echo "collector is still live; refusing concurrent resume" >&2; exit 1
      fi
      # Collector artifacts are root-owned inside the result mount. Archive the failure
      # sentinel and clear only the verified-stale lock through that same mount.
      docker run --rm --network none "${common_args[@]}" "$IMAGE" python3 -c \
        'import pathlib,sys,time; r=pathlib.Path("/results")/sys.argv[1]; f=r/"FAILED"; f.rename(r/f"FAILED.pre-resume.{int(time.time())}") if f.exists() else None; (r/".collector.lock").unlink(missing_ok=True)' \
        "$run_id"
    else
      [[ ! -e "$RLINF_RESULTS/$run_id/FAILED" ]] || { echo "FAILED exists; use grid-resume after inspection" >&2; exit 1; }
      [[ ! -e "$RLINF_RESULTS/$run_id/.collector.lock" ]] || { echo "collector lock exists; use grid-resume after verifying no live container" >&2; exit 1; }
    fi
    mode=parallel; [[ "$cmd" == grid-serial ]] && mode=serial
    workers="${RLINF_WORKERS:-2}"; node_ranks="${RLINF_NODE_RANKS:-0,0}"
    [[ "$mode" == serial ]] && { workers=1; node_ranks=0; }
    docker run --rm --name "$name" --network host "${common_args[@]}" "$IMAGE" \
      python3 /integration/src/grid_entry.py collect --config /integration/configs/grid_smoke.yaml \
      --output "/results/$run_id" --mode "$mode" -- \
      "run.id=$run_id" "runtime.workers=$workers" "runtime.node_ranks=[$node_ranks]" \
      "model_server.port=$PORT" "$@"
    ;;
  grid-consume)
    run_id="${2:?Usage: bash run.sh grid-consume RUN_ID}"
    validate_run_id "$run_id"
    docker run --rm --network host "${common_args[@]}" "$IMAGE" \
      python3 /integration/src/consume_collector.py "/results/$run_id" \
      --host 127.0.0.1 --port "$PORT"
    ;;
  grid-stop)
    run_id="${2:?Usage: bash run.sh grid-stop RUN_ID}"
    validate_run_id "$run_id"
    docker rm -f "rlinf-mibot-collector-${run_id}" 2>/dev/null || true
    docker rm -f "rlinf-mibot-model-${run_id}" 2>/dev/null || true
    ;;
  grid-compare)
    serial_id="${2:?Usage: bash run.sh grid-compare SERIAL_RUN PARALLEL_RUN}"
    parallel_id="${3:?Usage: bash run.sh grid-compare SERIAL_RUN PARALLEL_RUN}"
    validate_run_id "$serial_id"; validate_run_id "$parallel_id"
    # Result directories are created by root inside the rollout container, so run the
    # comparison through the same mounted image instead of writing as the host user.
    docker run --rm --network none "${common_args[@]}" "$IMAGE" \
      python3 /integration/src/compare_grid_runs.py "/results/$serial_id" "/results/$parallel_id" \
      --output "/results/$parallel_id/serial_vs_parallel.json"
    ;;
  grid-smoke)
    run_id="${2:?Usage: bash run.sh grid-smoke RUN_ID [OmegaConf overrides...]}"; shift 2
    validate_run_id "$run_id"
    bash preflight.sh
    "$0" model-start "$run_id"
    lifecycle_status=1
    cleanup() {
      if [[ "$lifecycle_status" -ne 0 ]]; then
        printf '{"status":"FAILED"}\n' > "$RLINF_RESULTS/$run_id/LIFECYCLE_FAILED"
      fi
      "$0" model-stop "$run_id" || true
    }
    trap cleanup EXIT INT TERM
    "$0" grid-parallel "$run_id" "$@"
    "$0" grid-consume "$run_id"
    printf '{"status":"DONE"}\n' > "$RLINF_RESULTS/$run_id/LIFECYCLE_DONE"
    lifecycle_status=0
    ;;
  *)
    printf '%s\n' 'Usage: bash run.sh {build|smoke|ppo-smoke|ode-eval|shell|model-start|model-stop|grid-parallel|grid-serial|grid-resume|grid-consume|grid-stop|grid-compare|grid-smoke}' >&2
    exit 2;;
esac
