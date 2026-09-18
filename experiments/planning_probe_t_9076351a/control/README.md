# Qwen planner cause-isolation control (t_574f0f33)

This directory is separate from the parent 12-case result. It runs direct greedy generation only: no parser repair, constrained decoding, fallback, planner implementation, or production-runtime change.

Controls:

1. `xiaomi_checkpoint_sanity`: canonical MiBot processor/VQA path on one preserved reset contact sheet, with three general image-QA prompts and the parent full-plan prompt.
2. `official_instruct`: official `Qwen/Qwen3-VL-4B-Instruct` processor/chat template on all three preserved reset contact sheets and the exact parent full-plan payload.

Run the controls sequentially on an idle `amp_csi` RTX3090 so only one model process exists. The generated artifacts record model revisions, processor/template details, image/prompt hashes, raw output, token IDs, first-token top-k, stop reason, and latency. The exact invocations are represented by the CLI arguments in `control_probe.py`; the preserved generation source commit is `1d8d20d3f0871567df6e961601d1da3e0547b8d8`.

Aggregate and verify:

    python3 experiments/planning_probe_t_9076351a/control/finalize.py \
      experiments/planning_probe_t_9076351a/control/results \
      --source-commit 1d8d20d3f0871567df6e961601d1da3e0547b8d8 \
      --parent-commit aa0807dc80aa0958a39ec315cb6bee9ea889c568
    python3 -m unittest experiments/planning_probe_t_9076351a/control/test_control.py
    python3 -m py_compile experiments/planning_probe_t_9076351a/control/*.py
    (cd experiments/planning_probe_t_9076351a/control/results && sha256sum -c artifacts.sha256)

Before and after GPU execution, verify zero compute processes, no experiment containers, and no listener on port 10087. Do not touch the v4 Arm A process/checkpoint.
