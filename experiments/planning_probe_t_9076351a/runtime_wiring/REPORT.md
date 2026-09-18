# Production pre-episode planner runtime wiring

Verdict: `RUNTIME_WIRING_READY`

This gate wires the validated one-call generic-Qwen full-plan contract into `architecture_smoke`. It does not claim closed-loop task accuracy or `PRODUCTION_READY`.

## Runtime contract

The canonical values live in `architecture_smoke/runtime_contract.py`: 20 Hz action consumption, 16 actions per chunk, 800 ms chunk deadline, 25 ms jitter limit, 1100 ms maximum source-observation age, and one queued chunk maximum.

After reset, `PreEpisodePlanner` sends one observation-only request to the generic Qwen endpoint. It accepts direct JSON only, requires exactly the registry sequence and exact argument fields, and materializes canonical five-field calls through the existing `SkillRegistry`. The episode loop validates the unchanged reset snapshot before invoking the executor. Malformed, unknown, wrong-sequence, extra-field, timeout/error, stale-response, and reset-divergence cases return with zero actions. There is no retry, JSON repair, `SequentialPlanner`, skill substitution, sequence correction, or fallback.

The combined server now optionally preloads pinned `Qwen/Qwen3-VL-4B-Instruct` alongside the Xiaomi policy model. Planner payloads use the generic processor/model; policy and verifier continue to use the existing Xiaomi endpoint paths. Runtime skill boundaries consume the frozen canonical plan and make zero planner calls.

## Verification

CPU:

- `python -m unittest discover -p 'test_*.py'`: 31/31 PASS.
- `py_compile`: runtime, server/backend, policy, and tests PASS.
- `git diff --check`: PASS.
- Contract integration: one planner call, first action only after validation, three deterministic boundaries, 48 mock actions, and boundary planner calls 0.
- Fail-closed matrix: malformed, unknown, wrong sequence, extra field, stale identity, reset divergence, timeout, and service error all execute action 0; fallback count 0.

RTX3090 `amp_csi` one-process smoke:

- Xiaomi policy and pinned generic Qwen planner loaded together in one Python process on one visible RTX3090.
- Planner calls: 1 pre-episode, 0 at runtime boundaries.
- Planner latency: 4086.464 ms (episode loop remained paused).
- Direct semantic JSON and strict registry canonicalization: PASS.
- Actions/boundaries: 48 / 3.
- Policy latency over three skill chunks: p50 276.177 ms, p95 279.794 ms, max 280.196 ms; all <=800 ms.
- Underflow/stale/deadline miss/producer error/RPC error/fallback/NaN: all 0.
- Maximum queue depth: 1 chunk.
- Maximum observed source age: 1099.968 ms (<1100 ms).
- Maximum measured action jitter: 0.172 ms (<25 ms).
- Cleanup: experiment containers 0, listeners on 10086/10087 0, compute processes 0, GPU 38 MiB / 0%.

Machine-readable evidence is in `results/report.json`; raw timing is in `results/timeline.csv` and `results/policy_latency.csv`.
