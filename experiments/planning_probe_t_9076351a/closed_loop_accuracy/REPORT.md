# Generic planner fixed-20 closed-loop task accuracy

Verdict: `TASK_ACCURACY_MEASURED`

This is a task-accuracy measurement, not a `PRODUCTION_READY` claim.

## Result

- Planner-valid: 20/20 (100%).
- Boundary reach: GRASP 20/20, MOVE 20/20, PLACE 20/20.
- Obs-only boundary success: GRASP 6/20, MOVE 8/20, PLACE 0/20.
- Official RoboCasa full-task success: 3/20 (15%); seeds [5005, 5011, 5017].
- Existing paired-seed baseline: 3/20; delta +0/20 (+0.0 pp).
- Paired discordance: baseline-only 2, planned-runtime-only 2; exact McNemar p=1.
- Interpretation: no aggregate improvement; the 20-seed result does not support an improvement claim.

## Mechanics hard gate

- Exactly one planner call per episode: True; runtime boundary calls 0.
- Fallback/repair/sequence correction/action-before-plan: 0/0/0/0.
- NaN/policy deadline/RPC-control/queue violations: 0/0/0/0.
- Latest-only superseded boundary observations: 4782 (expected coalescing, not queue violations).
- Completed scoring: 20/20.

## Latency

- Planner latency min/mean/max: 3248.8/3324.6/4222.6 ms.
- Policy forward p50/p95/max over 657 chunks: 350.1/401.0/449.6 ms (800 ms contract).

## Protocol and evidence

- Exact source commit: `45280448bd9f650c3cfd94f26939a8830374a05b`.
- Fixed target-split seeds: `5000..5019`; all failures remain in denominator 20.
- Generic `Qwen/Qwen3-VL-4B-Instruct@ebb281ec...` planner runs once from the reset 3-camera+proprio snapshot; direct raw JSON is in `planner_raw.jsonl` and each `seed*/result.json`.
- Xiaomi base policy and obs-only verifier receive no simulator predicates. Simulator predicates are read only after plan consumption for offline per-seed scoring.
- `per_seed.csv`, `episodes.json`, `seed*/trace.jsonl`, and `seed*/result.json` preserve reset hashes, canonical plans, transitions, actions, latencies, failure stages, and official outcomes.
- Three representative videos are listed in `videos/manifest.json`; no planning-failure video exists because all 20 plans were valid.
