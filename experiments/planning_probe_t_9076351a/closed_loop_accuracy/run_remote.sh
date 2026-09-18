#!/usr/bin/env bash
set -euo pipefail
base=/home/guest
pkg="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
parent="$base/robocasa-docker-t_9f03a613"
rlenv="$base/rl-env-t_4f3f2b20"
tools="$base/rollouts-xiaomi-t_4a072806/tools"
out="$pkg/output"
server=closed-loop-server-t_783b35bb
client=closed-loop-client-t_783b35bb
port=10086
mkdir -p "$out"

inventory() {
  {
    date -Is
    hostname
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
    docker ps --format '{{.Names}}|{{.Status}}|{{.Image}}'
    ss -ltn 2>/dev/null | grep -E ":(${port}|10087)\\b" || true
  } > "$1"
}

cleanup() {
  docker logs "$server" > "$out/server.log" 2>&1 || true
  docker rm -f "$client" "$server" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

inventory "$out/inventory_before.txt"
docker rm -f "$client" "$server" >/dev/null 2>&1 || true
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]; then
  echo "GPU compute process present; refusing to interfere" >&2
  exit 20
fi
if ss -ltn 2>/dev/null | grep -Eq ":${port}\\b"; then
  echo "port ${port} occupied" >&2
  exit 21
fi

docker run -d --name "$server" --gpus all --shm-size=2g \
  -e PYTHONPATH=/pkg:/rl_env/src -e HF_HOME=/root/.cache/huggingface \
  -v "$base/.cache/huggingface:/root/.cache/huggingface" \
  -v "$pkg/architecture_smoke:/pkg:ro" -v "$rlenv:/rl_env:ro" \
  -v "$parent/checkpoint:/checkpoint:ro" \
  xiaomi-cu121:t_9f03a613 \
  python3 /pkg/infer_verify_server.py --model /checkpoint --host 0.0.0.0 --port "$port" \
  > "$out/server.cid"

for _ in $(seq 1 180); do
  if docker logs "$server" 2>&1 | grep -q 'Infer+verify server running'; then
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$server"; then
    docker logs "$server" >&2
    exit 22
  fi
  sleep 5
done
docker logs "$server" 2>&1 | grep -q 'Infer+verify server running'
inventory "$out/inventory_loaded.txt"

docker run --rm --name "$client" --gpus all --shm-size=2g \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e PYTHONPATH=/pkg:/rl_env/src:/tools:/work \
  -v robocasa-assets-t_5af7225b:/opt/robocasa/robocasa/models/assets \
  -v "$parent:/work:ro" -v "$pkg/architecture_smoke:/pkg:ro" \
  -v "$pkg/experiment:/experiment:ro" -v "$rlenv:/rl_env:ro" \
  -v "$tools:/tools:ro" -v "$out:/output" \
  -v "$parent/checkpoint:/checkpoint:ro" \
  --network "container:$server" xiaomi-client:t_9f03a613 \
  python /experiment/run_closed_loop_accuracy.py \
    --out /output --source-commit 45280448bd9f650c3cfd94f26939a8830374a05b \
    --server-addr 127.0.0.1 --server-port "$port" --model-path /checkpoint \
    --split target --seeds 5000,5001,5002,5003,5004,5005,5006,5007,5008,5009,5010,5011,5012,5013,5014,5015,5016,5017,5018,5019 \
    --verifier-config /pkg/calib_out/balanced_t_a309678b/sequence_boundary_calibration.json \
    2>&1 | tee "$out/run.log"

cleanup
trap - EXIT INT TERM
inventory "$out/inventory_after.txt"
printf 'status=done timestamp=%s\n' "$(date -Is)" > "$out/DONE"
