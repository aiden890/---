# Single-GPU planner integration-readiness spike (t_56ccb5a7)

This throwaway probe loads the pinned Xiaomi action policy and the pinned generic Qwen planner in one Python process on one RTX3090. CUDA calls are serialized. It does not modify production runtime or the running v4 Arm A.

Measured sequence:

1. Idle-gate the amp_csi RTX3090.
2. Load Xiaomi policy and run two warmups plus 12 identical forwards.
3. Load Qwen planner into the same process and repeat the identical policy input.
4. Run one preserved seed-0 canonical planning case, then immediately run the policy again.
5. Record CUDA allocated/reserved/peak memory, driver free memory, wall latency, process inventory, raw semantic JSON, and canonical strict plan.

Final enum verdict: `INCONCLUSIVE`; production integration hard gate: `FAIL/NOT READY`. Co-residency itself passed the measured memory and throughput gates, but production code has no explicit policy-forward/control-loop deadline. The synchronous one-process design also pauses the action loop for the 3.30 s planner call. Under the task's fail-closed rule, this must not be integrated until a deadline and skill-boundary scheduling contract are defined and measured.

Exact remote invocation (after rsyncing this directory, `results/images/`, and `canonical_control/canonical_probe.py` to `/home/guest/experiments/planning_probe_t_9076351a/`):

    docker run --rm --name integration-readiness-t_56ccb5a7-r4 --gpus all --network host --shm-size=2g -e CUDA_VISIBLE_DEVICES=0 -e HF_HOME=/root/.cache/huggingface -v /home/guest/.cache/huggingface:/root/.cache/huggingface -v /home/guest/robocasa-docker-t_9f03a613/checkpoint:/checkpoint:ro -v /home/guest/experiments/planning_probe_t_9076351a:/experiment:ro -v /home/guest/experiments/planning_probe_t_9076351a/integration_readiness/remote_results_run4:/out xiaomi-cu121:t_9f03a613 python3 /experiment/integration_readiness/coresidency_probe.py --checkpoint /checkpoint --images /experiment/results/images --canonical-probe /experiment/canonical_control/canonical_probe.py --out /out --policy-repeats 12 --warmup 2 --max-new-tokens 256

Local verification:

    python3 -m unittest experiments/planning_probe_t_9076351a/integration_readiness/test_coresidency_probe.py
    python3 -m py_compile experiments/planning_probe_t_9076351a/integration_readiness/*.py
    (cd experiments/planning_probe_t_9076351a/integration_readiness/results && sha256sum -c artifacts.sha256)
    (cd experiments/planning_probe_t_9076351a/canonical_control/results && sha256sum -c artifacts.sha256)
    git diff --check -- experiments/planning_probe_t_9076351a/integration_readiness
