#!/usr/bin/env bash
# Run inside a provided SKKU GPU session. No inbound proxy is required for HF jobs.
set -euo pipefail
kit_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="${PI_CUP_ROOT:-$HOME/pi-cup-online-bc}"
mode="${1:-check}"
mkdir -p "$work_dir"
export HF_HOME="$work_dir/hf-cache"
export OPENPI_DATA_HOME="$work_dir/openpi-cache"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.6}"
export PYTHONPATH="$kit_dir/native-openpi/src:$kit_dir/native-openpi/packages/openpi-client/src:$kit_dir/src${PYTHONPATH:+:$PYTHONPATH}"
case "$mode" in
  setup)
    if ! command -v uv >/dev/null; then python3 -m pip install --user uv; export PATH="$HOME/.local/bin:$PATH"; fi
    test -d "$kit_dir/native-openpi" || { mkdir -p "$kit_dir/native-openpi"; tar -xzf "$kit_dir/vendor/openpi/coffee-openpi-portable.tgz" -C "$kit_dir/native-openpi"; }
    test -x "$work_dir/pi05-venv/bin/python" || uv venv --python 3.11 "$work_dir/pi05-venv"
    uv pip install --python "$work_dir/pi05-venv/bin/python" -r "$kit_dir/requirements/pi05.txt"
    test -x "$work_dir/transport-venv/bin/python" || uv venv --python 3.11 "$work_dir/transport-venv"
    uv pip install --python "$work_dir/transport-venv/bin/python" 'huggingface_hub>=2,<3'
    ;; 
  check)
    test -x "$work_dir/pi05-venv/bin/python"
    "$work_dir/pi05-venv/bin/python" -c 'import jax,torch,flax; from openpi.models.pi0_config import Pi0Config; print({"jax":jax.__version__,"flax":flax.__version__,"devices":[str(d) for d in jax.devices()]}); assert any(d.platform=="gpu" for d in jax.devices())'
    ;; 
  download)
    "$work_dir/pi05-venv/bin/python" "$kit_dir/src/download_pi_checkpoint.py" "$work_dir/checkpoints"
    ;; 
  validate)
    : "${HF_TOKEN_FILE:?Set HF_TOKEN_FILE to the private token file}"
    checkpoint="${PI_CUP_CHECKPOINT:-$work_dir/checkpoints/pi05_pretrain_human300/multitask_learning/75000}"
    "$work_dir/pi05-venv/bin/python" "$kit_dir/src/prepare_bootstrap.py" --root "$work_dir/validation-data" --transport-python "$work_dir/transport-venv/bin/python" --token-file "$HF_TOKEN_FILE"
    "$work_dir/pi05-venv/bin/python" "$kit_dir/tests/test_pipeline.py"
    "$work_dir/pi05-venv/bin/python" "$kit_dir/src/validate_dataset.py" "$work_dir/validation-data/data/pi05"
    "$work_dir/pi05-venv/bin/python" "$kit_dir/src/validate_dataset.py" "$work_dir/validation-data/data/xiaomi"
    "$work_dir/pi05-venv/bin/python" "$kit_dir/src/train_online_bc.py" --model pi05 --checkpoint "$checkpoint" --data "$work_dir/validation-data/data/pi05" "$work_dir/validation-data/data/xiaomi" --out "$work_dir/verification" --skills cup_placement --smoke --defer-inference
    "$work_dir/pi05-venv/bin/python" "$kit_dir/src/verify_inference.py" --model pi05 --checkpoint "$checkpoint" --data "$work_dir/validation-data/data/pi05" "$work_dir/validation-data/data/xiaomi" --out "$work_dir/verification"
    ;;
  start)
    "$work_dir/pi05-venv/bin/python" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["passed"] and r["frozen_backbone_unchanged"] and r["optimizer_resume_equal"]' "$work_dir/verification/verification.json"
    if test -f "$work_dir/learner.pid" && kill -0 "$(cat "$work_dir/learner.pid")" 2>/dev/null; then printf 'Existing learner is running.\n'; exit 0; fi
    : "${HF_TOKEN_FILE:?Set HF_TOKEN_FILE to the existing private token file, without pasting it into chat}"
    checkpoint="${PI_CUP_CHECKPOINT:-$work_dir/checkpoints/pi05_pretrain_human300/multitask_learning/75000}"
    test -d "$checkpoint/params"; test -d "$checkpoint/assets"; test -f "$HF_TOKEN_FILE"
    nohup "$work_dir/pi05-venv/bin/python" "$kit_dir/src/learner_service.py" --model pi05 --checkpoint "$checkpoint" --root "$work_dir/run" --run pi05-cup-online-bc-20261001 --transport-python "$work_dir/transport-venv/bin/python" --token-file "$HF_TOKEN_FILE" --bootstrap-models xiaomi pi05 --skills cup_placement --bootstrap-steps 0 --rounds 5 > "$work_dir/learner.log" 2>&1 < /dev/null &
    echo "$!" > "$work_dir/learner.pid"; printf 'LEARNER_PID=%s\nLOG=%s\n' "$(cat "$work_dir/learner.pid")" "$work_dir/learner.log"
    ;; 
  *) printf 'Usage: bash start-pi-cup-learner.sh setup|check|download|validate|start\n' >&2;exit 2;;
esac
