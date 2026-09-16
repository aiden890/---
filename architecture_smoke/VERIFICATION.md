# Harness-integrity verification + verification video (t_91bfee2b)

Follow-up to the adapter-free architecture smoke test (t_1d5b404f). Goal: prove
the skill-conditioned inference **harness** (the pipeline *around* the model) is
wired correctly, and separate **harness bugs** (must fix) from **model failures**
(base policy can't do a skill — OK, not a bug).

Pipeline under test:
Planner -> SkillCall(schema) -> instruction render -> base VLA (adapter OFF)
-> env.step -> Verifier(done_when/hold) -> ADVANCE/CONTINUE/REPLAN -> Planner.

## What was added

- `verify_traces.py` — audits the **recorded** rollout traces
  (`defined-instruction-rollouts/architecture_base_smoke/seed{0,1,2}/trace.jsonl`)
  against the 8 harness-integrity items, from real numbers (not mocks). Emits a
  per-item PASS/evidence table, a model-fail vs harness-bug split, and
  (`--json`) a structured block the tracking page consumes.
  Run: `python3 verify_traces.py` (pure JSONL audit, no GPU/numpy/sim).
- `check_architecture.py` — extended 15 -> **19** offline unit/integration checks:
  added `can_start`-failure, env-terminate-mid-skill, short-chunk-guard, and a
  deterministic double-run reproducibility check (the harness boundary paths the
  happy path never hits).
- `make_verification_video.py` — overlays the live harness state onto each frame
  of an existing `episode.mp4` (skill, planner-call, instruction, and the
  verifier verdict revealed **only** at the skill boundary; "executing …
  (verifier: CONTINUE)" during the run). Reuses the already-rendered frames +
  the trace; no simulator needed. Output is H.264 (browser-playable).

## Result (seeds 0,1,2, CloseBlenderLid, adapter disabled)

- 8/8 harness-integrity items **PASS**; **0 harness bugs** found.
- 22/22 trace-audit checks pass; 19/19 offline checks pass.
- 7 skill TIMEOUTs are all **clean model failures** (done_when unmet within
  `max_steps`, verifier correctly REPLANs with `hold` reset) — PLACE×4, GRASP×2,
  MOVE×1. These are base-policy limits, not harness bugs.

### The 8 items and how each was proven

1. **instruction 전달** — every `plan.rendered_instruction` matches `bindings.py`
   exactly; every `window` in a segment carries that same skill (no cross-talk).
2. **action 흐름** — window steps are strictly 16-strided; window count ==
   `ceil(elapsed/16)`; `chunk_len==executed==16` — no truncation/dup/reorder.
3. **verifier 판정** — ADVANCE fires at exactly `hold_steps` (GRASP20/MOVE3/PLACE1)
   with immediate SUCCESS; REPLAN only at full `max_steps`, `hold` reset to 0.
4. **handoff** — planner subgoal always follows the live predicate state it saw;
   `can_start=True` at every route; retry rationale present after a non-SUCCESS.
5. **adapter-free** — `adapter_mode=disabled`, `adapter_checkpoint=null` in
   episode_start, provenance, and *every* route+window record; asserts in code.
6. **predicate 소스** — digests carry only `skill_eval.Sim` predicate keys; no
   intra-step contradiction (single source of truth).
7. **경계값/에러** — per-skill steps never exceed `max_steps`; planner_calls<=cap;
   short-chunk/can_start/env-terminate covered in `check_architecture.py`.
   Note: `episode_budget` is a **planner-boundary soft cap** — the in-flight
   skill may overrun (seed2 792>600). This is defined behavior, not a bug.
8. **재현성** — `rollout.EvalClient` decodes actions by argmax (no
   multinomial/temperature RNG); identical seed re-run gives identical decisions
   (asserted by the double-run check).

## Tracking page

`tracking/index.html` smoke card gains a "하네스 무결성 정밀검증" section (8-item
table + verdict banner + model-fail split) and the verification video
(`media/architecture_smoke/architecture_verify_all.mp4`, 3-seed vstack);
per-episode drops now show the annotated `seed<N>_verify.mp4`. Data comes from
`smoke.json.harness_verification` (regenerate via
`verify_traces.py --json tracking/harness_verification.json`, then re-embed).
Served at http://100.86.183.64:8899/ (verified rendering + video decode in a real
browser).
