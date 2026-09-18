# Qwen3-VL planning capability probe (t_9076351a)

Production runtime is not modified. The probe invokes the real Xiaomi RoboCasa365 Qwen3-VL checkpoint without fallback and preserves direct raw generation.

Remote reproduction on an idle `amp_csi` RTX 3090:

1. Verify zero compute processes with `nvidia-smi` and no conflicting containers/port 10087.
2. Deploy this committed directory unchanged to `/home/guest/planning-probe-t_9076351a`.
3. Start one server container:

       docker run -d --name planning-probe-t_9076351a-server --gpus all --shm-size=2g \
         -v /home/guest/planning-probe-t_9076351a:/probe:ro \
         -v /home/guest/robocasa-docker-t_9f03a613/checkpoint:/checkpoint:ro \
         xiaomi-cu121:t_9f03a613 python3 /probe/probe_server.py --port 10087

4. After the `planning probe server listening` readiness line, run the client in the server network namespace:

       docker run --rm --name planning-probe-t_9076351a-client --gpus all --shm-size=2g \
         -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
         -v robocasa-assets-t_5af7225b:/opt/robocasa/robocasa/models/assets \
         -v /home/guest/robocasa-docker-t_9f03a613:/work:ro \
         -v /home/guest/rl-env-t_4f3f2b20:/rl_env:ro \
         -v /home/guest/planning-probe-t_9076351a:/probe:ro \
         -v /home/guest/planning-probe-t_9076351a-results:/output \
         -v /home/guest/robocasa-docker-t_9f03a613/checkpoint:/checkpoint:ro \
         --network container:planning-probe-t_9076351a-server \
         xiaomi-client:t_9f03a613 python3 /probe/probe.py --out /output --seeds 0,1,2 --port 10087

5. Remove both containers and verify port/GPU cleanup.

Local validation:

       python3 -m unittest experiments/planning_probe_t_9076351a/test_probe.py
       python3 -m py_compile experiments/planning_probe_t_9076351a/probe.py experiments/planning_probe_t_9076351a/probe_server.py

The preserved run is under `results/`. Recompute metrics from its immutable raw texts with:

       python3 experiments/planning_probe_t_9076351a/reanalyze.py experiments/planning_probe_t_9076351a/results
