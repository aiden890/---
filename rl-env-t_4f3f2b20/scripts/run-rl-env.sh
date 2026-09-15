#!/usr/bin/env bash
# Runner for the CloseBlenderLid skill-conditioned RL ENVIRONMENT (task t_4f3f2b20).
#
# Runs on v4. Reuses the verified images/checkpoint/assets and the ALREADY-RUNNING
# shared server (xiaomi-server-t_460aea68 on :10086) for baseline rollouts; it never
# stops or restarts that server. For RL (flow-SDE, log-prob) it starts a SEPARATE,
# short-lived RL server (src/rl_server.py) on :10087 in its own container and stops
# only that one.
#
# Layout on v4:
#   parent  = /home/v4/robocasa-docker-t_9f03a613           (has /checkpoint, /work rollout.py)
#   env     = /home/v4/rl-env-t_4f3f2b20                    (this package: src/, scripts/, configs/, tests/)
#   skill   = /home/v4/rollouts-xiaomi-t_4a072806/tools     (skill_eval.py, imported not copied)
#   out     = $env/results/<run>
#
# Usage:
#   bash run-rl-env.sh tests            # numpy+torch unit tests (client + server images)
#   bash run-rl-env.sh probe            # GPU: flow-SDE == checkpoint (eta=0) + log-prob sanity
#   bash run-rl-env.sh baseline <run>   # deterministic 3-skill oracle rollout via shared server
#   bash run-rl-env.sh rl <run>         # flow-SDE rollout w/ log-prob capture via RL server :10087
#   bash run-rl-env.sh rl-server-start|rl-server-stop
set -euo pipefail
parent=/home/v4/robocasa-docker-t_9f03a613
env=/home/v4/rl-env-t_4f3f2b20
skilltools=/home/v4/rollouts-xiaomi-t_4a072806/tools
shared_server=xiaomi-server-t_460aea68
rl_server=xiaomi-rl-server-t_4f3f2b20
server_image=xiaomi-cu121:t_9f03a613
client_image=xiaomi-client:t_9f03a613
assets_volume=robocasa-assets-t_5af7225b
rl_port=10087

need_shared() {
  docker ps --format '{{.Names}}' | grep -q "^${shared_server}$" || { echo "shared server ${shared_server} not running" >&2; exit 3; }
}

case "${1:-help}" in
  tests)
    echo "== numpy-only env components (client image) =="
    docker run --rm --entrypoint python -v "$env:/env:ro" "$client_image" /env/tests/test_env_components.py
    echo "== flow-SDE (client image has CPU torch) =="
    docker run --rm --entrypoint python -v "$env:/env:ro" "$client_image" /env/tests/test_flow_sde.py
    ;;
  probe)
    need_shared
    out="$env/results/probe"; mkdir -p "$out"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$out/gpu_before.txt"
    docker run --rm --name "xiaomi-rl-probe-t_4f3f2b20" --gpus all --shm-size=2g --network none \
      -v "$parent/checkpoint:/checkpoint:ro" -v "$env:/env:ro" -v "$out:/out" \
      --entrypoint python3 "$server_image" /env/scripts/sde_probe.py --checkpoint /checkpoint --out /out/probe_report.json \
      2>&1 | tee "$out/probe.log"
    ;;
  rl-server-start)
    docker ps --format '{{.Names}}' | grep -q "^${rl_server}$" && { echo "rl server already up"; exit 0; }
    docker run -d --name "$rl_server" --gpus all --network none --shm-size=2g \
      -v "$parent/checkpoint:/checkpoint:ro" -v "$env:/env:ro" \
      --entrypoint python3 "$server_image" /env/src/rl_server.py --model /checkpoint --host 127.0.0.1 --port $rl_port
    echo "started $rl_server on :$rl_port"
    ;;
  rl-server-stop)
    docker logs "$rl_server" > "$env/results/rl_server.log" 2>&1 || true
    docker stop "$rl_server" && docker rm "$rl_server" || true
    ;;
  baseline)
    need_shared
    run="${2:-run1}"; out="$env/results/baseline-$run"; mkdir -p "$out"
    docker ps --format '{{.Names}}' | grep -q '^xiaomi-client-rlenv' && { echo 'client already running' >&2; exit 4; }
    docker run --rm --name "xiaomi-client-rlenv-$run" --gpus all --shm-size=2g \
      -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
      -v "$assets_volume:/opt/robocasa/robocasa/models/assets" -v "$parent:/work:ro" \
      -v "$skilltools:/skill_eval_tools:ro" -v "$env:/env:ro" -v "$out:/output" \
      -v "$parent/checkpoint:/checkpoint:ro" --network "container:$shared_server" \
      --entrypoint python "$client_image" /env/src/rl_rollout.py \
      --out /output --mode baseline --planner oracle --reward-mode simulator \
      --seed "${3:-9}" --eta 0.0 "${@:4}" 2>&1 | tee "$out/rollout.log"
    ;;
  rl)
    # RL server must be up on :10087; the client shares its network namespace.
    docker ps --format '{{.Names}}' | grep -q "^${rl_server}$" || { echo "start rl server first: bash $0 rl-server-start" >&2; exit 5; }
    run="${2:-run1}"; out="$env/results/rl-$run"; mkdir -p "$out"
    docker run --rm --name "xiaomi-client-rlenv-$run" --gpus all --shm-size=2g \
      -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
      -v "$assets_volume:/opt/robocasa/robocasa/models/assets" -v "$parent:/work:ro" \
      -v "$skilltools:/skill_eval_tools:ro" -v "$env:/env:ro" -v "$out:/output" \
      -v "$parent/checkpoint:/checkpoint:ro" --network "container:$rl_server" \
      --entrypoint python "$client_image" /env/src/rl_rollout.py \
      --out /output --mode rl --planner oracle --reward-mode simulator \
      --seed "${3:-9}" --eta "${4:-0.6}" --rl-server-port $rl_port --capture-branches "${@:5}" \
      2>&1 | tee "$out/rollout.log"
    ;;
  *)
    echo "Usage: bash run-rl-env.sh {tests|probe|baseline <run> [seed]|rl <run> [seed] [eta]|rl-server-start|rl-server-stop}" >&2
    exit 2;;
esac
