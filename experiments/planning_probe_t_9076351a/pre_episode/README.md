# Pre-episode full-plan scheduler hard gate

This directory is a throwaway measurement harness and contract. It does not alter or deploy the production Xiaomi server, client, or RoboCasa evaluator.

Contract: models are preloaded and warmed; reset is acquired with the simulator/action loop paused; the generic planner runs exactly once and emits the complete three-skill plan; direct JSON and strict deterministic canonical validation must pass; only then may the first policy chunk and first action be produced. Runtime skill boundaries consume the preserved canonical plan without planner calls.

The harness explicitly tests 20 action/s, a 16-action chunk, an 800 ms chunk-generation deadline, at most one buffered chunk, 25 ms action schedule jitter, and 1100 ms maximum source-observation age. Invalid JSON, invalid canonical plans, wrong sequence, stale reset observations, or reset divergence execute zero actions and use no fallback.

Production source facts are preserved in `source_contract.json`. In particular, `--replan-steps=16` controls consumption, while the production loop has no sleep, target control rate, or action deadline; `--video-fps=20` is encoder metadata only. Therefore this artifact validates a candidate scheduler contract, not production integration or deployment.

Local verification:

    python3 -m unittest experiments/planning_probe_t_9076351a/pre_episode/test_pre_episode_probe.py
    python3 -m py_compile experiments/planning_probe_t_9076351a/pre_episode/*.py
    (cd experiments/planning_probe_t_9076351a/pre_episode && sha256sum -c artifacts.sha256)
