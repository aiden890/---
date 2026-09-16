#!/usr/bin/env bash
# run-vlm-inference.sh (task t_fc5e73d5) -- obs-only VLM(Qwen3-VL) verifier inference.
#
# Runs the full skill-conditioned architecture for CloseBlenderLid with the
# skill-termination judge = the policy's OWN frozen Qwen3-VL VQA (obs-only:
# 3-cam + 14D proprio, no privileged sim predicate). ONE model load serves both
# the base-policy action forward and the op=vlm_score VQA forward.
#
# Designed for amp_csi (free 24GB 3090). Reuses the SAME checkpoint / images /
# assets volume / parent rollout+skill_eval tools as the other cards.
#
# Usage (on amp_csi):
#   bash run-vlm-inference.sh "0,1,2,3,4"          # seeds
set -euo pipefail

seeds="${1:-0,1,2,3,4}"

base=/home/guest
parent="$base/robocasa-docker-t_9f03a613"
rlenv="$base/rl-env-t_4f3f2b20"
pkg="$base/architecture-smoke-t_fc5e73d5"
tools="$base/rollouts-xiaomi-t_4a072806/tools"
assets_volume=robocasa-assets-t_5af7225b
server=vlm-infer-server-t_fc5e73d5
client=vlm-infer-client-t_fc5e73d5
port=10086
out="$pkg/out"
mkdir -p "$out"

# clean any stale containers from a previous run
docker rm -f "$server" "$client" >/dev/null 2>&1 || true

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$out/gpu_before.txt"

# ---- 1. GPU server: one model load, base actions + op=vlm_score -------------
docker run -d --name "$server" --gpus all --shm-size=2g \
  -e PYTHONPATH=/pkg:/rl_env/src \
  -v "$pkg/architecture_smoke:/pkg:ro" \
  -v "$rlenv:/rl_env:ro" \
  -v "$parent/checkpoint:/checkpoint:ro" \
  xiaomi-cu121:t_9f03a613 \
  python /pkg/infer_verify_server.py --model /checkpoint --host 0.0.0.0 --port "$port"

# wait for the model to load (server prints "Model loaded." then "running on")
echo "waiting for server to load model..."
for i in $(seq 1 120); do
  if docker logs "$server" 2>&1 | grep -q "server running on"; then
    echo "server ready"; break
  fi
  if ! docker ps --format '{{.Names}}' | grep -q "^$server$"; then
    echo "SERVER DIED:"; docker logs "$server" 2>&1 | tail -30; exit 3
  fi
  sleep 5
done

# ---- 2. sim client: rollout + obs-only VLM verifier ------------------------
docker run --rm --name "$client" --gpus all --shm-size=2g \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e PYTHONPATH=/pkg:/rl_env/src:/tools:/work \
  -v "$assets_volume:/opt/robocasa/robocasa/models/assets" \
  -v "$parent:/work:ro" \
  -v "$pkg/architecture_smoke:/pkg:ro" \
  -v "$rlenv:/rl_env:ro" \
  -v "$tools:/tools:ro" \
  -v "$out:/output" -v "$parent/checkpoint:/checkpoint:ro" \
  --network "container:$server" xiaomi-client:t_9f03a613 \
  python /pkg/run_vlm_inference.py --out /output --seeds "$seeds" \
    --model-path /checkpoint --server-addr 127.0.0.1 --server-port "$port" \
    2>&1 | tee "$out/run.log"

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$out/gpu_after.txt"

# ---- 3. stop the server ----------------------------------------------------
docker rm -f "$server" >/dev/null 2>&1 || true

# ---- 4. overlay: VLM-verifier annotated videos (cv2 in the client image) ---
# no GPU / server needed; pure cv2 + trace.jsonl. One per seed that produced a video.
for sd in "$out"/seed*; do
  [ -d "$sd" ] || continue
  sn=$(basename "$sd" | sed 's/seed//')
  [ -f "$sd/episode.mp4" ] || continue
  docker run --rm --name "vlm-overlay-t_fc5e73d5-$sn" \
    -e PYTHONPATH=/pkg \
    -v "$pkg/architecture_smoke:/pkg:ro" -v "$out:/output" \
    xiaomi-client:t_9f03a613 \
    python /pkg/make_vlm_verification_video.py --root /output --seed "$sn" \
      --out "/output/seed${sn}/episode_vlm_verify.mp4" \
      >> "$out/overlay.log" 2>&1 || echo "overlay seed$sn failed" >> "$out/overlay.log"
done

echo "done. results in $out"
