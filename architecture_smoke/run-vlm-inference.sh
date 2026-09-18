#!/usr/bin/env bash
# Canonical amp_csi launcher for the one-model/two-queue Skill VLA runtime.
#
# Commands:
#   run-vlm-inference.sh run [seed_csv]   full run with atomic DONE/FAILED
#   run-vlm-inference.sh start            start server and enforce readiness
#   run-vlm-inference.sh health           strict health readback
#   run-vlm-inference.sh readiness        strict readiness readback
#   run-vlm-inference.sh stop             stop runtime containers and verify cleanup
#   run-vlm-inference.sh rollback         stop and restore the previous run pointer
set -euo pipefail

command="${1:-run}"
seeds="${2:-0,1,2}"
base="${BASE_DIR:-/home/guest}"
pkg="${PKG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
parent="${PARENT_DIR:-$base/robocasa-docker-t_9f03a613}"
rlenv="${RL_ENV_DIR:-$base/rl-env-t_4f3f2b20}"
tools="${TOOLS_DIR:-$base/rollouts-xiaomi-t_4a072806/tools}"
assets_volume="${ASSETS_VOLUME:-robocasa-assets-t_5af7225b}"
port="${PORT:-10086}"
server="skill-vla-runtime-server"
client="skill-vla-runtime-client"
state_dir="$pkg/.runtime"
runs_dir="$pkg/runtime_runs"
state_file="$state_dir/active.env"
mkdir -p "$state_dir" "$runs_dir"

atomic_text() {
  local target="$1" text="$2" tmp="${1}.tmp.$$"
  printf '%s\n' "$text" > "$tmp"
  mv -f "$tmp" "$target"
}

container_running() {
  docker ps --format '{{.Names}}' | grep -qx "$1"
}

inventory() {
  local target="$1"
  {
    printf 'timestamp=%s\n' "$(date -Is)"
    printf 'host=%s\n' "$(hostname)"
    printf '%s\n' '=== gpu ==='
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader
    printf '%s\n' '=== gpu processes ==='
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
    printf '%s\n' '=== containers ==='
    docker ps --format '{{.Names}}|{{.Status}}|{{.Ports}}'
    printf '%s\n' '=== ports ==='
    ss -ltnp 2>/dev/null | grep -E ":(${port}|8899)\\b" || true
    printf '%s\n' '=== relevant processes ==='
    ps -u "$(id -un)" -o pid,lstart,args --sort=start_time \
      | grep -E '(stage1|run-train|grpo|skill-vla|vlm|infer_verify|run_vlm)' \
      | grep -v grep || true
  } > "$target"
}

assert_idle() {
  if container_running "$server" || container_running "$client"; then
    echo "existing Skill VLA workload is live; refusing duplicate start" >&2
    return 1
  fi
  if ss -ltn 2>/dev/null | grep -Eq ":${port}\\b"; then
    echo "port $port is already listening; refusing duplicate start" >&2
    return 1
  fi
  if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]; then
    echo "GPU already has a compute workload; refusing to interfere" >&2
    return 1
  fi
}

strict_health() {
  local output="${1:-}"
  local args=(--host 127.0.0.1 --port "$port" --timeout 5 --strict)
  if [ -n "$output" ]; then
    local tmp="${output}.tmp.$$"
    docker exec "$server" python3 /pkg/runtime_probe.py "${args[@]}" > "$tmp"
    mv -f "$tmp" "$output"
  else
    docker exec "$server" python3 /pkg/runtime_probe.py "${args[@]}"
  fi
}

wait_ready() {
  local output="$1"
  for _ in $(seq 1 120); do
    if ! container_running "$server"; then
      docker logs "$server" >&2 || true
      echo "server exited before readiness" >&2
      return 1
    fi
    if docker logs "$server" 2>&1 | grep -q 'server running on'; then
      if strict_health "$output" >/dev/null 2>&1; then
        return 0
      fi
    fi
    sleep 5
  done
  echo "server readiness timeout" >&2
  return 1
}

start_server() {
  local out="$1"
  assert_idle
  docker run -d --name "$server" --gpus all --shm-size=2g \
    -e PYTHONPATH=/pkg:/rl_env/src \
    -v "$pkg:/pkg:ro" -v "$rlenv:/rl_env:ro" \
    -v "$parent/checkpoint:/checkpoint:ro" \
    xiaomi-cu121:t_9f03a613 \
    python3 /pkg/infer_verify_server.py --model /checkpoint --host 0.0.0.0 --port "$port" \
    > "$out/server.cid"
  docker inspect "$server" > "$out/server_inspect.json"
  wait_ready "$out/health_ready.json"
  docker logs "$server" > "$out/server.log" 2>&1
}

stop_runtime() {
  docker rm -f "$client" "$server" >/dev/null 2>&1 || true
  if container_running "$client" || container_running "$server"; then
    echo "runtime container cleanup failed" >&2
    return 1
  fi
  if ss -ltn 2>/dev/null | grep -Eq ":${port}\\b"; then
    echo "runtime port cleanup failed: $port still listening" >&2
    return 1
  fi
}

write_provenance() {
  local out="$1"
  cp "$pkg/source_manifest.json" "$out/source_manifest.json"
  sha256sum "$out/source_manifest.json" > "$out/source_manifest.sha256"
  find "$parent/checkpoint" -maxdepth 1 -type f -print0 \
    | sort -z | xargs -0 sha256sum > "$out/checkpoint.sha256"
  {
    printf 'command=%q run %q\n' "$0" "$seeds"
    printf 'source_commit=%s\n' "$(python3 "$pkg/read_manifest_commit.py" "$pkg/source_manifest.json")"
    printf 'server_image=%s\n' 'xiaomi-cu121:t_9f03a613'
    printf 'client_image=%s\n' 'xiaomi-client:t_9f03a613'
    printf 'seeds=%s\nport=%s\n' "$seeds" "$port"
  } > "$out/run_manifest.txt"
}

run_all() {
  local run_id out previous=""
  run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
  out="${OUT_DIR:-$runs_dir/$run_id}"
  mkdir -p "$out"
  [ ! -L "$runs_dir/current" ] || previous="$(readlink "$runs_dir/current")"
  atomic_text "$out/PREVIOUS" "$previous"
  ln -sfn "$run_id" "$runs_dir/current.next"
  mv -Tf "$runs_dir/current.next" "$runs_dir/current"
  atomic_text "$state_file" "RUN_ID=$run_id
OUT=$out
SERVER=$server
CLIENT=$client
PORT=$port
PREVIOUS=$previous"

  failed() {
    local rc=$?
    trap - ERR INT TERM
    docker logs "$server" > "$out/server.log" 2>&1 || true
    strict_health "$out/health_failure.json" >/dev/null 2>&1 || true
    stop_runtime || true
    inventory "$out/inventory_after_failure.txt" || true
    atomic_text "$out/FAILED" "status=failed rc=$rc timestamp=$(date -Is)"
    if [ -n "$previous" ]; then
      ln -sfn "$previous" "$runs_dir/current.next"
      mv -Tf "$runs_dir/current.next" "$runs_dir/current"
    else
      rm -f "$runs_dir/current"
    fi
    exit "$rc"
  }
  trap failed ERR INT TERM

  inventory "$out/inventory_before.txt"
  assert_idle
  write_provenance "$out"
  start_server "$out"
  if [ "${FAIL_AFTER_READY:-0}" = 1 ]; then
    echo "injected failure after readiness" >&2
    false
  fi

  docker run --rm --name "$client" --gpus all --shm-size=2g \
    -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    -e PYTHONPATH=/pkg:/rl_env/src:/tools:/work \
    -v "$assets_volume:/opt/robocasa/robocasa/models/assets" \
    -v "$parent:/work:ro" -v "$pkg:/pkg:ro" -v "$rlenv:/rl_env:ro" \
    -v "$tools:/tools:ro" -v "$out:/output" \
    -v "$parent/checkpoint:/checkpoint:ro" \
    --network "container:$server" xiaomi-client:t_9f03a613 \
    python /pkg/run_vlm_inference.py --out /output --seeds "$seeds" \
      --model-path /checkpoint --server-addr 127.0.0.1 --server-port "$port" \
      --replan-steps 16 --vlm-min-interval 16 \
      2>&1 | tee "$out/run.log"

  strict_health "$out/health_final.json" >/dev/null
  docker logs "$server" > "$out/server.log" 2>&1
  python3 "$pkg/runtime_metrics.py" --run-dir "$out" \
    --health "$out/health_final.json" --output "$out/metrics.json" > "$out/metrics.log"
  sha256sum "$out/config.json" > "$out/config.sha256"
  stop_runtime
  inventory "$out/inventory_after.txt"
  sha256sum "$out/metrics.json" "$out/summary.json" "$out/run.log" "$out/server.log" \
    > "$out/artifacts.sha256"
  atomic_text "$out/DONE" "status=done timestamp=$(date -Is) run_id=$run_id"
  rm -f "$state_file"
  trap - ERR INT TERM
  printf '%s\n' "$out"
}

case "$command" in
  run)
    run_all
    ;;
  start)
    run_id="${RUN_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
    out="${OUT_DIR:-$runs_dir/$run_id}"
    mkdir -p "$out"
    inventory "$out/inventory_before.txt"
    write_provenance "$out"
    start_server "$out"
    atomic_text "$state_file" "RUN_ID=$run_id
OUT=$out
SERVER=$server
CLIENT=$client
PORT=$port
PREVIOUS="
    printf '%s\n' "$out"
    ;;
  health|readiness)
    strict_health
    ;;
  stop)
    stop_runtime
    rm -f "$state_file"
    ;;
  rollback)
    previous=""
    if [ -f "$state_file" ]; then
      # shellcheck disable=SC1090
      source "$state_file"
      previous="${PREVIOUS:-}"
    elif [ -L "$runs_dir/current" ]; then
      current="$(readlink "$runs_dir/current")"
      [ ! -f "$runs_dir/$current/PREVIOUS" ] || previous="$(cat "$runs_dir/$current/PREVIOUS")"
    fi
    stop_runtime
    if [ -n "$previous" ]; then
      ln -sfn "$previous" "$runs_dir/current.next"
      mv -Tf "$runs_dir/current.next" "$runs_dir/current"
    else
      rm -f "$runs_dir/current"
    fi
    atomic_text "$state_dir/ROLLBACK" "status=rolled_back timestamp=$(date -Is) previous=$previous"
    rm -f "$state_file"
    ;;
  *)
    echo "usage: $0 {run [seed_csv]|start|health|readiness|stop|rollback}" >&2
    exit 2
    ;;
esac