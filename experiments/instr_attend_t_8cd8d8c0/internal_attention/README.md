# Internal attention experiment

This directory extends `experiments/instr_attend_t_8cd8d8c0`; it does not fork the policy or simulator project.

## Outputs

- `raw/{state}__{instruction}.npz`: VLM/DiT attention summaries and action output for all 12 conditions.
- `raw/{state}__{instruction}.tokens.json`: exact sequence, instruction/image indices, and GRASP/MOVE/PLACE semantic-token mapping.
- `raw/manifest.jsonl`: shape/head/layer/timestep checks and condition-level means.
- `analysis/combined_summary.{json,csv}`: join with `../run2/attendance_stats.json`.
- `analysis/*.png`: layer/timestep/head heatmaps and attention-vs-action-sensitivity scatter.
- `analysis/REPORT.ko.md`: Korean findings, agreement/disagreement, and causal caveats.
- `RUN_PROVENANCE.json`: exact commits, source/checkpoint hashes, images, GPU host, and snapshot steps.

## Reproduction

Deploy committed source from the central repository to `amp_csi` only after confirming the RTX3090 is idle and has no compute process:

    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
    rsync -a --exclude __pycache__ experiments/instr_attend_t_8cd8d8c0/internal_attention/ \
      amp_csi:/home/guest/experiments/instr_attend_t_8cd8d8c0/internal_attention/source/

Start exactly one model process:

    docker run -d --name xiaomi-attn-server-c47fb375-v2 --gpus all --shm-size=2g \
      -e MIBOT_SERVER_SEED=7 \
      -v /home/guest/robocasa-docker-t_9f03a613/checkpoint:/checkpoint:ro \
      -v /home/guest/experiments/instr_attend_t_8cd8d8c0/internal_attention/source:/attention:ro \
      -v /home/guest/experiments/instr_attend_t_8cd8d8c0/internal_attention/raw:/output \
      xiaomi-cu121:t_9f03a613 python3 /attention/attention_server.py \
      --model /checkpoint --out /output --host 0.0.0.0 --port 10086

Run the simulator client on the server container network (mounts are the same as the parent experiment):

    docker run --name xiaomi-attn-client-c47fb375 --gpus all --shm-size=2g \
      -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
      -v robocasa-assets-t_5af7225b:/opt/robocasa/robocasa/models/assets \
      -v /home/guest/robocasa-docker-t_9f03a613:/work:ro \
      -v /home/guest/rollouts-xiaomi-t_4a072806/tools:/skilltools:ro \
      -v /home/guest/experiments/instr_attend_t_8cd8d8c0/tools:/tools:ro \
      -v /home/guest/experiments/instr_attend_t_8cd8d8c0/internal_attention/source:/attention:ro \
      -v /home/guest/experiments/instr_attend_t_8cd8d8c0/internal_attention/raw:/output \
      -v /home/guest/robocasa-docker-t_9f03a613/checkpoint:/checkpoint:ro \
      --network container:xiaomi-attn-server-c47fb375-v2 \
      xiaomi-client:t_9f03a613 python /attention/internal_probe.py \
      --out /output --seed 9 --gen-attempts 6 --gen-horizon 900

Stop the experiment-owned server, verify GPU release, then aggregate in the client image:

    docker stop xiaomi-attn-server-c47fb375-v2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
    docker run --name internal-attn-analysis-c47fb375-v2 \
      -v /home/guest/experiments/instr_attend_t_8cd8d8c0/internal_attention/source:/attention:ro \
      -v /home/guest/experiments/instr_attend_t_8cd8d8c0/internal_attention/raw:/raw:ro \
      -v /home/guest/experiments/instr_attend_t_8cd8d8c0/internal_attention/analysis:/analysis \
      -v /home/guest/experiments/instr_attend_t_8cd8d8c0/attendance_stats.json:/attendance.json:ro \
      xiaomi-client:t_9f03a613 python /attention/analyze_internal_attention.py \
      --input /raw --attendance /attendance.json --out /analysis

CPU/synthetic aggregation smoke test:

    python /attention/test_analysis_smoke.py
