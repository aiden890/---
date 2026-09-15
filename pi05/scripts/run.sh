#!/usr/bin/env bash
# pi0.5 (openpi) RoboCasa365 eval orchestrator (v4).
# Mirrors the GR00T card contract (t_ff372eea): separate checkpoint dir + output,
# never touches xiaomi resources, PandaOmron embodiment, split=pretrain for
# apples-to-apples with Xiaomi/GR00T. Public pi0.5 RoboCasa365 checkpoint — inference only.
#
# openpi is JAX-based, so the POLICY SERVER runs in its own vendor image
# (openpi-server, built from robocasa-benchmark/openpi serve_policy.Dockerfile), while
# the eval CLIENT reuses the proven groot-eval sim image + openpi-client (pi05-client).
# Two containers on one GPU, connected over a private docker network.
#
#   bash pi05/scripts/run.sh build-client     # build the sim+openpi_client eval image
#   bash pi05/scripts/run.sh download          # fetch pi0.5 RoboCasa365 ckpt from HF
#   bash pi05/scripts/run.sh eval [N]          # serve pi0.5 + run CloseBlenderLid eval (N ep)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$(dirname "$here")")"   # repo root when run locally; on v4 = WD
cd "$root"

WD=${WD:-$root}
server_image=${SERVER_IMAGE:-openpi-server-rc:t_ee702b9c}
client_image=${CLIENT_IMAGE:-pi05-client:t_ee702b9c}
client_base=${CLIENT_BASE_IMAGE:-groot-eval:t_ff372eea}
asset_volume=${ROBOCASA_ASSET_VOLUME:-robocasa-assets-t_9f03a613}
ckpt_dir=${PI05_CKPT_DIR:-$WD/checkpoints/pi05_robocasa365}
out_dir=${PI05_OUT_DIR:-$WD/output}
hf_repo="robocasa/robocasa365_checkpoints"
hf_subpath="pi05_pretrain_human300/multitask_learning/75000"
config="pi05_pretrain_human300"
net=${PI05_NET:-pi05-eval-net}
port=${PI05_PORT:-8000}

mkdir -p "$WD/logs" "$out_dir"

case "${1:-help}" in
  build-client)
    docker build -f pi05/docker/Dockerfile.client \
      --build-arg BASE_IMAGE="$client_base" \
      -t "$client_image" "$WD" 2>&1 | tee "$WD/logs/build-client.log"
    ;;

  download)
    # JAX/orbax checkpoint subtree (params ocdbt + assets/norm_stats.json + metadata).
    mkdir -p "$ckpt_dir"
    docker run --rm -e HF_HUB_ENABLE_HF_TRANSFER=0 \
      -v "$ckpt_dir:/ckpt" "$server_image" \
      /.venv/bin/python - "$hf_repo" "$hf_subpath" <<'PY' 2>&1 | tee "$WD/logs/download.log"
import sys
from huggingface_hub import snapshot_download
repo, sub = sys.argv[1], sys.argv[2]
p = snapshot_download(repo_id=repo, allow_patterns=[f"{sub}/*"], local_dir="/ckpt")
print("downloaded to", p, "subpath", sub)
PY
    echo "checkpoint -> $ckpt_dir/$hf_subpath"
    ;;

  eval)
    n_ep=${2:-50}
    replan=${PI05_REPLAN:-16}
    split=${PI05_SPLIT:-pretrain}
    task=${PI05_TASK:-CloseBlenderLid}
    docker network create "$net" 2>/dev/null || true
    # --- start policy server (JAX pi0.5) ---
    docker rm -f pi05-server 2>/dev/null || true
    # Launch the server with the venv python directly (NOT `uv run`, which re-syncs the
    # venv and clobbers our robocasa/mujoco pins). openpi is src-layout, so PYTHONPATH
    # points at src + the openpi-client package.
    docker run -d --name pi05-server --network "$net" --gpus all \
      -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
      -e HF_HUB_OFFLINE=1 \
      -e PYTHONPATH=/app/src:/app/packages/openpi-client/src \
      -w /app \
      -v "$WD/openpi:/app" \
      -v "$ckpt_dir/$hf_subpath:/ckpt:ro" \
      "$server_image" \
      /.venv/bin/python scripts/serve_policy.py --port "$port" \
        policy:checkpoint --policy.config="$config" --policy.dir=/ckpt \
      > "$WD/logs/serve.cid" 2>&1
    echo "server starting; waiting for readiness..."
    # serve_policy logs "Creating server (host..." only AFTER the JAX model is loaded.
    for i in $(seq 1 120); do
      if docker logs pi05-server 2>&1 | grep -qiE "Creating server \(host"; then echo "server ready"; break; fi
      if ! docker ps --format '{{.Names}}' | grep -q '^pi05-server$'; then echo "SERVER DIED"; docker logs --tail 40 pi05-server 2>&1; exit 1; fi
      sleep 5
    done
    docker logs --tail 20 pi05-server 2>&1 | tee "$WD/logs/serve.log"
    # --- run eval client (sim) ---
    set +e
    docker run --rm --name pi05-client --network "$net" --gpus all --shm-size=2g \
      -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
      -e HF_HUB_OFFLINE=1 \
      -v "$asset_volume:/opt/robocasa/robocasa/models/assets:ro" \
      -v "$out_dir:/output" \
      -v "$WD/pi05:/pi05:ro" \
      "$client_image" \
      python3 /pi05/scripts/eval_closeblenderlid.py \
        --task "$task" --split "$split" --n_episodes "$n_ep" \
        --replan_steps "$replan" --host pi05-server --port "$port" \
        --video_dir /output/CloseBlenderLid \
      2>&1 | tee "$WD/logs/eval-$task.log"
    rc=${PIPESTATUS[0]}
    set -e
    docker rm -f pi05-server 2>/dev/null || true
    exit "$rc"
    ;;

  *)
    printf 'Usage: bash pi05/scripts/run.sh {build-client|download|eval [N]}\n' >&2
    exit 2
    ;;
esac
