#!/usr/bin/env bash
set -euo pipefail
# Resolve the project root (the Docker build context that holds output/, configs/,
# scripts/, docker/). Works whether this script is invoked through the root compat
# symlink (./run.sh) or directly as scripts/run.sh.
here="$(cd "$(dirname "$0")" && pwd)"
if [ "$(basename "$here")" = scripts ]; then root="$(dirname "$here")"; else root="$here"; fi
cd "$root"
image=${ROBOCASA_IMAGE:-robocasa-sim:t_04bf4fb7}
volume=${ROBOCASA_ASSET_VOLUME:-robocasa-assets-t_04bf4fb7}
dockerfile=docker/sim/Dockerfile
case "${1:-smoke}" in
  build)
    docker build -f "$dockerfile" -t "$image" .
    ;;
  assets)
    # Named-volume initialization preserves the asset metadata shipped in Git.
    printf 'y\n' | docker run --rm -i -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
      -v "$volume:/opt/robocasa/robocasa/models/assets" "$image" \
      python -m robocasa.scripts.download_kitchen_assets --type "${@:2}"
    ;;
  smoke|cpu|physics)
    mode=${1:-smoke}
    mkdir -p "output/$mode"
    opts=()
    render=(--render)
    if [[ $mode == smoke ]]; then
      opts+=(--gpus all -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl)
      render+=(--require-gpu)
    else
      opts+=(-e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa)
    fi
    if [[ $mode == physics ]]; then render=(); fi
    docker run --rm "${opts[@]}" --shm-size=2g \
      -v "$volume:/opt/robocasa/robocasa/models/assets" \
      -v "$PWD/output/$mode:/output" "$image" \
      python /app/smoke_test.py "${render[@]}" "${@:2}"
    ;;
  *) printf 'Usage: bash run.sh {build|assets all|assets tex fixtures_lw|smoke|cpu|physics}\n' >&2; exit 2 ;;
esac
