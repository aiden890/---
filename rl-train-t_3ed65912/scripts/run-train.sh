#!/usr/bin/env bash
# Runner for the CloseBlenderLid skill-conditioned GRPO TRAINING loop (task t_3ed65912).
#
# Runs on v4. Reuses the verified images/checkpoint/assets and the env card
# (rl-env-t_4f3f2b20, mounted read-only as /rl_env = single source of truth for the
# flow-SDE sampler, reward, skill FSM). NEVER stops/restarts the shared inference
# server xiaomi-server-t_460aea68. Starts its OWN short-lived GRPO trainer server
# (src/grpo_trainer_server.py) on :10088 in its own container and stops only that one.
#
# Feasibility (scripts/train_feasibility_probe.py): AdamW OOMs beside the shared
# server; grad-checkpoint + SGD fits (group4=12.6GB, group6=13.1GB peak). So the
# canonical pilot = group 4, SGD, gradient checkpointing.
#
# Layout on v4:
#   parent = /home/v4/robocasa-docker-t_9f03a613   (/checkpoint, /work rollout.py)
#   rlenv  = /home/v4/rl-env-t_4f3f2b20            (env card: src reused by import)
#   train  = /home/v4/rl-train-t_3ed65912          (this card: trainer + loop)
#   skill  = /home/v4/rollouts-xiaomi-t_4a072806/tools (skill_eval.py, imported)
set -euo pipefail
parent=/home/v4/robocasa-docker-t_9f03a613
rlenv=/home/v4/rl-env-t_4f3f2b20
train=/home/v4/rl-train-t_3ed65912
skilltools=/home/v4/rollouts-xiaomi-t_4a072806/tools
shared_server=xiaomi-server-t_460aea68
trainer=xiaomi-grpo-trainer-t_3ed65912
server_image=xiaomi-cu121:t_9f03a613
client_image=xiaomi-client:t_9f03a613
assets_volume=robocasa-assets-t_5af7225b
port=10088

case "${1:-help}" in
  probe)
    out="$train/results"; mkdir -p "$out"
    docker run --rm --gpus all --shm-size=2g --network none \
      -v "$parent/checkpoint:/checkpoint:ro" -v "$rlenv:/rl_env:ro" -v "$train:/train" \
      --entrypoint python3 "$server_image" \
      /train/scripts/train_feasibility_probe.py "${@:2}" --out /train/results/train_feasibility.json
    ;;
  trainer-start)
    docker ps --format '{{.Names}}' | grep -q "^${trainer}$" && { echo "trainer already up"; exit 0; }
    docker run -d --name "$trainer" --gpus all --network none --shm-size=2g \
      -v "$parent/checkpoint:/checkpoint:ro" -v "$rlenv:/rl_env:ro" -v "$train:/train" \
      --entrypoint python3 "$server_image" \
      /train/src/grpo_trainer_server.py --model /checkpoint --host 127.0.0.1 --port $port "${@:2}"
    echo "started $trainer on :$port (args: ${*:2})"
    ;;
  trainer-stop)
    docker logs "$trainer" > "$train/results/trainer_server.log" 2>&1 || true
    docker stop "$trainer" && docker rm "$trainer" || true
    ;;
  audit-p0)
    # GPU regression for the P0 audit fixes (adapter-active-in-eval + checkpoint roundtrip).
    # Own short-lived container; loads the model in-process (no trainer server, no shared server).
    out="$train/results/audit"; mkdir -p "$out"
    docker run --rm --gpus all --shm-size=2g --network none \
      -v "$parent/checkpoint:/checkpoint:ro" -v "$rlenv:/rl_env:ro" -v "$train:/train" -v "$out:/out" \
      --entrypoint python3 "$server_image" \
      /train/scripts/audit_p0_verify.py --out /out/audit_p0_verify.json "${@:2}"
    ;;
  audit-branch)
    # AUDIT item #4 integration test (client image = sim+assets, networked to trainer).
    # Needs the trainer server up (for actions). Verifies real simulator state restore +
    # shared-prefix group branching.
    docker ps --format '{{.Names}}' | grep -q "^${trainer}$" || { echo "start trainer first: bash $0 trainer-start" >&2; exit 5; }
    out="$train/results/audit"; mkdir -p "$out"
    docker run --rm --name "xiaomi-client-audit-branch" --gpus all --shm-size=2g \
      -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
      -v "$assets_volume:/opt/robocasa/robocasa/models/assets" -v "$parent:/work:ro" \
      -v "$skilltools:/skill_eval_tools:ro" -v "$rlenv:/rl_env:ro" -v "$train:/train" -v "$out:/out" \
      -v "$parent/checkpoint:/checkpoint:ro" --network "container:$trainer" \
      --entrypoint python "$client_image" /train/scripts/audit_branch_verify.py \
      --trainer-port $port --out /out/audit_branch_verify.json "${@:2}" 2>&1 | tee "$out/audit_branch.log"
    ;;
  train)
    docker ps --format '{{.Names}}' | grep -q "^${trainer}$" || { echo "start trainer first: bash $0 trainer-start" >&2; exit 5; }
    run="${2:-run1}"; out="$train/results/$run"; mkdir -p "$out"
    docker run --rm --name "xiaomi-client-grpo-$run" --gpus all --shm-size=2g \
      -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
      -v "$assets_volume:/opt/robocasa/robocasa/models/assets" -v "$parent:/work:ro" \
      -v "$skilltools:/skill_eval_tools:ro" -v "$rlenv:/rl_env:ro" -v "$train:/train" -v "$out:/out" \
      -v "$parent/checkpoint:/checkpoint:ro" --network "container:$trainer" \
      --entrypoint python "$client_image" /train/src/grpo_train_loop.py \
      --out /out --trainer-port $port "${@:3}" 2>&1 | tee "$out/train.log"
    ;;
  sft)
    # Goal-3 SFT client (CFM per-skill LoRA). Uses the SERVER image (torch+hf+torchvision
    # +pyarrow for demo parquet/video decode; no sim needed) networked to the trainer.
    # The trainer must be started with the SFT hyperparams (e.g. --optimizer adamw --lr 1e-4
    # --train-mode adapter_only --grad-checkpoint). Subcommand form:
    #   bash run-train.sh sft <out-subdir> --op overfit --skill GRASP_HANDLE --arm nl_plus_skill_id ...
    docker ps --format '{{.Names}}' | grep -q "^${trainer}$" || { echo "start trainer first: bash $0 trainer-start" >&2; exit 5; }
    run="${2:-sft}"; out="$train/results/skill_sft/$run"; mkdir -p "$out"
    # Trainer runs --network none (GRPO isolation), so this client cannot pip-install or
    # hf-download at run time. Deps (pyarrow/av/hf_hub) are pre-staged in $train/pylibs and
    # the ONLY needed dataset files (CBL parquet + 3 shared video chunks) in $train/hf_cache;
    # HF_HUB_OFFLINE=1 makes hf_hub_download resolve from that local_dir. Re-stage with
    # scripts/skill_sft_stage_data.sh if pylibs/hf_cache are missing.
    docker run --rm --name "xiaomi-sft-$run" --gpus all --shm-size=2g \
      -v "$parent/checkpoint:/checkpoint:ro" -v "$rlenv:/rl_env:ro" -v "$train:/train" -v "$out:/out" \
      -e HF_HUB_OFFLINE=1 -e HF_HOME=/train/hf_home --network "container:$trainer" \
      --entrypoint bash "$server_image" -lc \
      "PYTHONPATH=/train/pylibs:/train/src python3 /train/scripts/skill_sft_train.py \
       --model /checkpoint --port $port --cache /train/hf_cache --out /out ${*:3}" 2>&1 | tee "$out/sft.log"
    ;;
  *)
    echo "Usage: bash run-train.sh {probe|trainer-start [server args]|trainer-stop|audit-p0|audit-branch|train <run> [args]|sft <run> [sft args]}" >&2
    exit 2;;
esac
