#!/usr/bin/env bash
set -euo pipefail
parent=/home/v4/robocasa-docker-t_9f03a613
root=/home/v4/rollouts-xiaomi-t_5af7225b
server=xiaomi-server-t_5af7225b
mkdir -p "$root"
case "${1:-}" in
start)
 docker run -d --name "$server" --gpus all --network none --shm-size=2g -v "$parent/checkpoint:/checkpoint:ro" -v "$parent:/work:ro" xiaomi-cu121:t_9f03a613 python3 /work/upstream/deploy/server.py --model /checkpoint --host 127.0.0.1 --port 10086
 ;;
episode)
 category=$2; task=$3; seed=$4; attempt=$5
 taskset=${category//-/_}
 dir="$root/$category/$attempt"
 test ! -e "$dir"
 mkdir -p "$dir"
 printf '{"category":"%s","task":"%s","base_seed":%s,"attempt":"%s","split":"target","task_set":"%s","replan_steps":16,"obs_history":4,"obs_interval":2,"crop_ratio":0.95,"video_stride":2,"video_fps":20,"horizon":"official task default"}\n' "$category" "$task" "$seed" "$attempt" "$taskset" > "$dir/config.json"
 docker run --rm --name "xiaomi-client-t_5af7225b-$attempt" --gpus all --shm-size=2g -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics -v robocasa-assets-t_5af7225b:/opt/robocasa/robocasa/models/assets -v "$parent:/work:ro" -v "$root:/output" -v "$parent/checkpoint:/checkpoint:ro" --network "container:$server" xiaomi-client:t_9f03a613 python /work/rollout.py --model-path /checkpoint --server-addr 127.0.0.1 --server-port 10086 --split target --task-set "$taskset" --task-name "$task" --num-trials 1 --seed "$seed" --save-videos --save-root-dir "/output/$category/$attempt" --run-id rollout 2>&1 | tee "$dir/rollout.log"
 ;;
stop)
 docker logs "$server" > "$root/server.log" 2>&1
 docker stop "$server"
 ;;
*) printf '%s\n' 'Usage: bash run-t_5af7225b.sh start|stop|episode CATEGORY TASK BASE_SEED UNIQUE_ATTEMPT'; exit 2;;
esac
