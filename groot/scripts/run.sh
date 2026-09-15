#!/usr/bin/env bash
# GR00T N1.5 RoboCasa365 eval orchestrator (v4). Mirrors the xiaomi run.sh contract:
# separate checkpoint volume/dir, task-scoped output, never touches xiaomi resources.
#
#   bash groot/scripts/run.sh build          # build the combined eval image
#   bash groot/scripts/run.sh download        # fetch gr00t_n1-5 multitask checkpoint from HF
#   bash groot/scripts/run.sh smoke           # import + gym.make(CloseBlenderLid) render smoke
#   bash groot/scripts/run.sh eval-full [N]    # full-task CloseBlenderLid eval, N episodes (default 50)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$(dirname "$here")")"   # repo root (…/robocasa-docker)
cd "$root"

image=${GROOT_IMAGE:-groot-eval:t_ff372eea}
base=${GROOT_BASE_IMAGE:-xiaomi-cu121:t_9f03a613}
asset_volume=${ROBOCASA_ASSET_VOLUME:-robocasa-assets-t_9f03a613}
ckpt_dir=${GROOT_CKPT_DIR:-$root/groot/checkpoints/gr00t_n1-5_multitask}
out_dir=${GROOT_OUT_DIR:-$root/groot/results}
hf_repo="robocasa/robocasa365_checkpoints"
hf_subpath="gr00t_n1-5/multitask_learning/checkpoint-120000"

mkdir -p "$root/groot/logs" "$out_dir"

case "${1:-help}" in
  build)
    docker build -f groot/docker/Dockerfile --build-arg BASE_IMAGE="$base" \
      -t "$image" . 2>&1 | tee "$root/groot/logs/build.log"
    ;;

  download)
    # Weights only (2 safetensors shards + config + experiment_cfg metadata); skip the
    # 8.5GB optimizer.pt — inference does not need it.
    mkdir -p "$ckpt_dir"
    docker run --rm -e HF_HUB_ENABLE_HF_TRANSFER=0 \
      -v "$ckpt_dir:/ckpt" "$image" python3 - "$hf_repo" "$hf_subpath" <<'PY' 2>&1 | tee "$root/groot/logs/download.log"
import sys
from huggingface_hub import hf_hub_download
repo, sub = sys.argv[1], sys.argv[2]
files = [
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "experiment_cfg/metadata.json",
]
for f in files:
    p = hf_hub_download(repo_id=repo, filename=f"{sub}/{f}", local_dir="/ckpt")
    print("downloaded", p)
print("DONE")
PY
    # Flatten: point /ckpt/model at the nested checkpoint-120000 dir for a clean --model_path.
    ln -sfn "$ckpt_dir/$hf_subpath" "$ckpt_dir/model"
    echo "checkpoint model dir -> $ckpt_dir/model"
    ;;

  smoke)
    docker run --rm --gpus all --network none --shm-size=2g \
      -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
      -v "$asset_volume:/opt/robocasa/robocasa/models/assets:ro" \
      -v "$root/groot:/groot:ro" "$image" \
      python3 /groot/scripts/smoke_eval.py 2>&1 | tee "$root/groot/logs/smoke.log"
    ;;

  eval-full)
    n_ep=${2:-50}
    n_action=${GROOT_N_ACTION_STEPS:-16}
    split=${GROOT_SPLIT:-target}
    task_set=${GROOT_TASK_SET:-atomic_seen}
    model_path=${GROOT_MODEL_PATH:-/ckpt/model}
    # Combined server+client (one process, one GPU). --split target matches Xiaomi's
    # RoboCasa365 target50 protocol; CloseBlenderLid is in atomic_seen.
    docker run --rm --gpus all --shm-size=2g \
      -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
      -e HF_HUB_OFFLINE=1 \
      -v "$asset_volume:/opt/robocasa/robocasa/models/assets:ro" \
      -v "$ckpt_dir:/ckpt:ro" -v "$out_dir:/output" \
      -v "$root/groot:/groot:ro" "$image" \
      python3 /opt/Isaac-GR00T/scripts/run_eval.py \
        --model_path "$model_path" \
        --embodiment_tag new_embodiment \
        --data_config panda_omron \
        --task_set "$task_set" \
        --split "$split" \
        --video_dir /output/CloseBlenderLid \
        --n_episodes "$n_ep" --n_envs 1 --n_action_steps "$n_action" \
      2>&1 | tee "$root/groot/logs/eval-full.log"
    ;;

  *)
    printf 'Usage: bash groot/scripts/run.sh {build|download|smoke|eval-full [N]}\n' >&2
    exit 2
    ;;
esac
