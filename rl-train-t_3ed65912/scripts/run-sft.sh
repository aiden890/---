#!/usr/bin/env bash
# Skill-SFT stage runner (task t_e5cd5736), extending rl-train-t_3ed65912.
# GPU-free data steps run in an ephemeral CPU container (no GR00T/GPU interference);
# GPU steps (SFT gates, rollout) reuse the trainer container path from run-train.sh.
#
# Subcommands:
#   data           Goal-1: download + segment demos, build split/mask/manifest (CPU only)
#   conditioning   Goal-2: print the per-arm conditioning + run the unit test (CPU only)
#   (GPU steps sft-probe/sft-overfit/sft-train/sft-val/sft-rollout-gate are added in the
#    SFT build step; they reuse run-train.sh trainer-start/-stop.)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # rl-train-t_3ed65912 root
IMG="${SFT_CPU_IMAGE:-xiaomi-cu121:t_9f03a613}"
DATA_OUT="$HERE/data/skill_sft"
EVID_OUT="$HERE/results/skill_sft"
CMD="${1:-help}"; shift || true

cpu_run() {
  # run a data script in an ephemeral CPU container with pyarrow available.
  # HF cache goes to a container-internal /hfcache (NOT the mounted tree) so no
  # root-owned files are left on the host; only OUT/DATA_OUT (host-owned) are written.
  local script="$1"
  docker run --rm --cpus "${SFT_CPUS:-4}" --user "$(id -u):$(id -g)" \
    -v "$HERE:/work" -e OUT=/work/results/skill_sft -e DATA_OUT=/work/data/skill_sft \
    -e HF_HOME=/tmp/hf -e SFT_HF_CACHE=/tmp/hf/cache \
    --entrypoint bash "$IMG" -lc \
    "python3 -m pip install --quiet --target /tmp/pylibs pyarrow 2>/dev/null; PYTHONPATH=/tmp/pylibs python3 /work/scripts/$script"
}

case "$CMD" in
  data)
    mkdir -p "$DATA_OUT" "$EVID_OUT"
    cpu_run skill_sft_inspect_dataset.py
    cpu_run skill_sft_extract_cbl.py
    cpu_run skill_sft_segmentation_probe.py
    cpu_run skill_sft_finger_timeline.py
    cpu_run skill_sft_segment_and_manifest.py
    echo "== data manifest =="; cat "$DATA_OUT/data_manifest.json"
    ;;
  conditioning)
    python3 "$HERE/src/skill_sft_conditioning.py"
    python3 "$HERE/tests/test_skill_sft_conditioning.py"
    ;;
  help|*)
    sed -n '2,14p' "${BASH_SOURCE[0]}"
    ;;
esac
