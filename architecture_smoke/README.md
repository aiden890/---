# Adapter-free full-architecture inference smoke test (CloseBlenderLid)

Kanban task **t_1d5b404f**. Verifies the full skill-conditioned inference
architecture is wired correctly **before any adapter training**. This is an
*architecture integration test*, not a policy benchmark: base-policy skill
failures are expected and do NOT fail the smoke test.

## Data flow

```
                 +-----------------------------------------------------------+
 goal + obs ---> | OraclePlanner (planner.py)   *scripted stub, NOT learned* |
                 |   PlannerContext -> SkillCall(name, args)                  |
                 +----------------------------+------------------------------+
                                              | typed SkillCall
                                              v
        +---------------------------- ExecutionManager (executor.py) --------------------------+
        |  registry.validate_call  ->  contract.render_instruction  ->  loop:                  |
        |     env.build_policy_input -> BasePolicyClient.infer (policy.py, ADAPTER DISABLED)    |
        |       -> env.step (environment.py, RoboCasa) -> PredicateVerifier.update (verifier.py)|
        |     until ADVANCE / REPLAN / env-terminate                                           |
        |  -> SkillResult(status, ...)                                                          |
        +----------------------------------------+---------------------------------------------+
                                                 | SkillResult
                                                 v
                          back to OraclePlanner (advance / retry / replan)
```

Loop repeats until `official_check_success` (task predicate) or the episode
budget is exhausted. Every planner boundary and skill window is written to
`seed<N>/trace.jsonl` by `trace.py`.

## Module boundaries (dependencies point inward through `schemas.py`)

| file | responsibility | must NOT do |
|------|----------------|-------------|
| `schemas.py` | typed dataclasses: PlannerContext, SkillCall, SkillContract, PolicyInput/Output, VerificationResult, SkillResult | any I/O |
| `bindings.py` | CloseBlenderLid task bindings + exact defined instructions | generic orchestration |
| `skills.py` | SkillRegistry: catalog, arg validation, instruction render | model inference |
| `planner.py` | OraclePlanner (scripted stub) PlannerContext -> SkillCall | simulator / policy calls |
| `verifier.py` | PredicateVerifier: done_when + budget -> ADVANCE/CONTINUE/RETRY/REPLAN | select next skill |
| `executor.py` | ExecutionManager state machine: validate->render->infer->step->verify | task-specific predicates |
| `policy.py` | BasePolicyClient (wraps rollout.EvalClient); MockPolicy | load any adapter |
| `environment.py` | RoboCasaEnvironment (wraps skill_eval.Sim + rollout); MockEnvironment | control decisions |
| `trace.py` | JSONL trace + frame metadata | control decisions |
| `episode.py` | outer planner<->manager loop | task predicates / policy internals |
| `run_architecture_smoke.py` | thin CLI composition | behavior |

Reuse-by-import (never forked): `rollout.py` (base VLA EvalClient, history
sampling, video frames) and `skill_eval.py` `Sim` (predicate single source of
truth) from the parent `robocasa-docker-t_9f03a613`.

## Adapter-free guarantees

* `BasePolicyClient` refuses any `adapter_mode` other than `disabled/base_only`
  and holds `adapter_checkpoint = None` as an invariant.
* Every `PolicyInput` is asserted adapter-disabled at infer time; the executor
  additionally asserts `PolicyOutput.adapter_checkpoint is None` each window.
* `config.json` + every `trace.jsonl` record the policy provenance
  (`model_path`, `adapter_mode=disabled`, `adapter_checkpoint=null`).

## Planner honesty

`OraclePlanner` is a **scripted stub** honoring the planner *interface*
(PlannerContext in, schema-valid SkillCall out, no sim/policy access). No
skill-conditioned VLM planner has been trained. Never report its output as a
learned planner. `kind="oracle_stub"` is tagged into every trace record.

## Run

Offline checks (any CPU box, no GPU/numpy/sim):

```
python3 check_architecture.py    # 15/15 must pass
```

Full GPU smoke test (inside the xiaomi-client container, VLA server up):

```
bash run-architecture-smoke.sh          # seeds 0,1,2 -> <out>/seed{0,1,2}/
```

Artifacts land in
`/home/aiden/Desktop/lab/robot/defined-instruction-rollouts/architecture_base_smoke/`
(`config.json`, `summary.json`, `seed<N>/{trace.jsonl,summary.json,scene.json,episode.mp4}`).

## Obs-only skill-termination verifier (t_d4268a2e)

The smoke-test verifier (`verifier.py`) judges skill termination from the
simulator's **privileged** predicates (`skill_eval.Sim.predicates()` →
`lid_on_blender`, `official_check_success`, `lid_pos` …) — object-pose / fixture
state a real robot cannot observe, so it cannot transfer to hardware. The
obs-only redesign judges termination from **only what the robot receives**
(3-cam images + 14-D proprio), with the policy's own frozen Qwen3-VL backbone as
the completion judge and a proprio event-gate for a realistic VLM cadence.

| file | responsibility |
|------|----------------|
| `obs_verifier.py` | `ObsInput` (frozen, images+proprio only) + `assert_obs_only` guard rejecting every privileged sim predicate; `ProprioGate` (settle/gripper events → VLM cadence gate + baseline); `ObsVLMVerifier` (VLM-judged completion, hysteresis latch, realistic cadence, timeout→REPLAN) |
| `vlm_backends.py` | `MockVLMBackend` (tests) + `QwenVLMScorerBackend` (policy's own Qwen3-VL; reuses `vlm_scorer` VQA prompt + logit→prob math, single source of truth; 3-cam horizontal concat) |
| `test_obs_verifier.py` | 25 unit tests: obs-only schema guard, camera-key guard, proprio gate, hysteresis latch, cadence gating, timeout→REPLAN, mock e2e |
| `validate_obs_verifier.py` | stream-replay agreement harness (obs-verifier ADVANCE vs sim ground-truth success step; precision/recall/timing-offset) |
| `probe_vlm_discrimination.py` | real-VLM discrimination probe on gt-labeled frames (done vs not-done separation) |
| `make_report.py` / `build_obsv_tracking.py` | assemble `REPORT/obs_verifier.{md,json}` and `tracking/obs_verifier.json` from artifacts |

**Obs-only invariant:** the runtime judge input is images+proprio only; every
privileged sim predicate key is rejected in code (`assert_obs_only`, re-checked
each `ObsVLMVerifier.update`). Privileged predicates are used ONLY as offline
supervision labels in the validation harness. Run:

```
python3 test_obs_verifier.py                    # 25/25 must pass (CPU, no GPU)
# real VLM (inside xiaomi-client container, CPU or GPU):
python3 probe_vlm_discrimination.py --rollouts-root <dir> --specs '<sub>:<skill>,...' --out out_probe
python3 validate_obs_verifier.py --rollouts-root <dir> --glob 'rand10_grasp_seed*' --skill grasp --backend qwen --out out_qwen_grasp
```

See `REPORT/obs_verifier.md` for the validation findings (GRASP obs-judgeable,
PLACE not single-frame; hysteresis latch load-bearing; 433 s/forward on CPU).
