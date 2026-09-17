# Skill VLA Overall Framework — current/target readiness audit

이 문서는 UI 배포안이 아니라 승인 전 architecture audit이다. 현재 코드로 확인된 구현과 목표 topology를 분리한다. `tracking/index.html`, live 8899 페이지, 학습 프로세스는 변경하지 않았다.

## 결론

현재 코드는 아래 두 가지를 각각 입증한다.

1. base Xiaomi policy와 observation-only verifier가 **한 물리 GPU의 한 Qwen3-VL/Xiaomi model instance**를 공유할 수 있다. `InferVerifyServer`가 모델을 한 번만 load하고 policy forward와 `vlm_score` forward를 하나의 CUDA lock으로 직렬화한다.
2. scripted planner → typed `SkillCall` → manager → policy → RoboCasa → obs-only verifier → `SkillResult`의 실험용 synchronous loop가 존재한다.

그러나 목표인 `Target async deployment`는 아직 구현·배포되지 않았다. 특히 async verifier client, latest-only queue(size 1), policy-priority scheduler, request identity/stale rejection, RPC timeout/retry, endpoint request type, 배포 health gate와 rollback automation이 없다. 현재 verifier 호출은 manager control thread를 동기 block하며, 두 server thread가 동일 lock을 먼저 얻는 순서대로 실행될 뿐 policy 우선순위는 없다.

따라서 UI에는 현재 상태를 `deployed async architecture`로 표현하면 안 된다. 목표도 반드시 `Proposed deployment topology / Target async deployment`로 표시해야 한다.

## 1. Current implementation topology

```text
SequentialPlanner (scripted fixed order; obs-only result consumption)
  -> SkillCall{name,args}
  -> ExecutionManager (single synchronous control thread)
       -> BasePolicyClient/EvalClient --socket--> InferVerifyServer
       |                                  |
       |                                  +-- one loaded Xiaomi/Qwen3-VL model
       |                                  +-- one non-priority threading.Lock
       |                                  +-- base action handler
       |                                  +-- vlm_score handler
       |
       -> RoboCasaEnvironment.step(one decoded 12-D action)
       -> ObsVLMVerifier.update(obs)
            -> RemoteVLMScorerBackend --separate socket, synchronous RPC--+
       -> SuccessGate.observe(recorded frames) --synchronous RPC----------+
       -> SkillResult
  -> planner retry/advance

Simulator predicates
  -> predicate-mode runtime path when obs_only=False
  -> final/offline labels and reports when obs_only=True
```

### Current facts that must remain explicit

- Planner: `SequentialPlanner` is a fixed `GRASP_OBJECT -> MOVE_OBJECT -> PLACE_OBJECT` stub. It is **not a learned VLM planner** and does not use goal text or scene observation to select skills (`planner.py:67-107`).
- Shared model: `InferVerifyServer` loads one `AutoModel` once and uses the same object for policy and `model.vlm` scoring (`infer_verify_server.py:43-55,81-112`).
- Shared CUDA: both request types are serialized by one ordinary `threading.Lock`; this is mutual exclusion, not priority scheduling (`infer_verify_server.py:47,105-112`).
- Runtime verifier input: three cameras plus 14-D proprio; forbidden simulator keys are rejected (`obs_verifier.py:53-140`, `environment.py:81-93`).
- Policy input/output: the policy path receives three camera histories, a 14-D state padded into the model's 60-D state tensor, and the rendered skill instruction. The decoded client consumes the first 12 action dimensions; manager executes 16 actions per default chunk (`xiaomi-cu121/rollout.py:32-37,59-78,135-185`, `policy.py:47-55`).
- Control authority: current `ObsVLMVerifier` returns `ADVANCE/CONTINUE/REPLAN` locally, and `ExecutionManager` maps those to results (`obs_verifier.py:330-418`, `executor.py:126-168`).
- SuccessGate: separate strict reporting gate exists for GRASP and PLACE, but it runs synchronously from the manager and is not a distinct endpoint request contract (`executor.py:74-78,108-147`, `success_gate.py:128-194`).
- Offline GT: in obs-only mode the planner receives an empty predicate map; official predicates are read at episode end for evaluation (`episode.py:18-23,32-43,69-86`). The resulting report still includes `final_predicates`, so consumers must keep those fields in an evaluation-only plane.

## 2. Target topology (not deployed)

```text
PLANNING PLANE
  Task / Goal Input
    -- task goal + scene observation + plan history --> VLM Planner Service
  Skill Contract Registry
    -- schema + args + can_start + done + instruction + max_steps --> Planner/Manager
  VLM Planner
    -- SkillCall{name,args,instruction,contract,budget} --> Ordered Plan State / Manager

CONTROL HOST / EXECUTION PLANE
  Execution Manager / Router
    -- policy request --> Shared GPU Inference Server.Policy API (high priority)
    <-- 16-step action chunk --
    -- one action per 20 Hz step --> RoboCasa / Robot Control
    <-- cameras + proprio14 + done/trunc --
    -- immutable latest observation --> Async Verifier Client

  Async Verifier Client
    - queue capacity 1; replace pending request with newest frame
    - at most one verifier RPC in flight
    - request key: episode_id + skill_id + observation_step + request_id + request_kind
    - reject response if episode/skill changed or response step is older than accepted state
    - own timeout, cancellation/drop, hysteresis, and ADVANCE/CONTINUE/RETRY/REPLAN
    -- low-priority async request --> Shared GPU Inference Server.Verifier API
    <-- P(yes) + view + model latency + echoed request metadata --

SHARED GPU INFERENCE PLANE — SAME physical GPU, SAME model instance
  Shared GPU Inference Server
    - one Xiaomi/Qwen3-VL model load
    - high-priority Policy API queue
    - low-priority Verifier API queue
    - explicit scheduler; one CUDA forward at a time
    - pending verifier may be dropped before start; policy cannot be queued behind pending verifier work

VERIFICATION
  Frozen Qwen3-VL Verifier logical endpoint (inside the shared server)
  Endpoint SuccessGate request_kind=endpoint (same endpoint/model, stricter local reduction)

TELEMETRY
  Manager -> Trace: SkillCall, action window, verifier request/result/drop/stale/timeout,
                    transition, queue wait, model latency, policy latency

EVALUATION ONLY
  Simulator GT / official predicates --dashed, offline only--> calibration/evaluation/tracking
  No edge from GT to planner, manager, policy, or runtime verifier.
```

### Target scheduling semantics

- Policy and verifier are independent application-level lanes but share one GPU/model; CUDA execution remains serialized.
- Fetch a policy chunk first. After the chunk is available, submit at most one background verifier request while its 16 actions execute.
- Default target cadence is one verifier request per 16-step chunk (16 / 20 Hz = 0.8 s), plus one rising-edge candidate-stop request. Do not query every settled step.
- A latest-only pending slot removes backlog; replacement happens only before server execution starts. An in-flight CUDA forward is not preempted.
- Measured sequence runs show warm VLM forwards around 80–96 ms, with one 169.94 ms warm-up sample (`calib_out/balanced_t_a309678b/e2e_sequence/seed40..42/summary.json`, `e2e_optimized/summary.json`). At an 0.8 s cadence, an 85 ms forward occupies about 10.6% GPU duty. Async control does not make this cost zero; a collision with an already-running verifier can delay policy by one verifier forward, and warm-up can be longer.

## 3. Interface audit

| Producer | Consumer | Current payload/cadence | Current mode | Target contract | Readiness |
|---|---|---|---|---|---|
| Task/scene | Planner | `PlannerContext{goal,predicates/catalog,last_result,budget,calls}`; obs-only planner ignores goal/scene | local sync | goal + scene obs + plan history | NOT READY |
| Registry | Planner/Manager | Python `bindings.CONTRACTS`; runtime does not load `tracking/inference.json` | local sync | one canonical versioned registry with schema, conditions, instruction, budget | PARTIAL |
| Planner | Manager | `SkillCall{name,args}` | local sync | include/resolve immutable instruction, contract revision, budget, call/skill id | PARTIAL |
| Manager | Policy | `PolicyInput{instruction,state_history,image_history,adapter_mode,None}` per empty action plan | socket sync | request id + 3-camera history + proprio14 + instruction + model/checkpoint revision | PARTIAL |
| Policy | Manager | decoded first 12 dims, manager truncates to 16 steps | socket sync | 16-step chunk + raw/model shape + latency + revision | PARTIAL |
| Manager | Robot | one action per `env.step`; no wall-clock 20 Hz pacing in harness | local sync | one action / 20 Hz with deadline/overrun telemetry | PARTIAL |
| Robot | Manager | obs + done/trunc; verifier adapter emits 3 cams + proprio14 | local sync | timestamped immutable observation with episode/step identity | PARTIAL |
| Manager/client | Verifier endpoint | VQA tensors + question only | synchronous socket RPC | latest frame(s), skill/question, episode_id, skill_id, request_id, observation_step, kind | NOT READY |
| Verifier endpoint | Client | `{prob,question}` | synchronous socket RPC | probability, view, model/queue latency, echoed identity and model revision | NOT READY |
| Manager | Planner | `SkillResult` with status/reason/steps and report fields | local sync | explicit `SUCCESS/TIMEOUT/FAILED`, retryability, final obs ref | PARTIAL |
| Manager | Trace | JSONL plan/route/window/frame/verifier/result | local sync | add request lifecycle, stale/drop/timeout, queue and inference timing | PARTIAL |
| Simulator GT | Offline eval | final and trace/report labels; isolated from obs-only planner input | offline by convention + `obs_only` branch | typed evaluation-only sink and assertion of no runtime subscribers | PARTIAL |

## 4. Module ownership and gaps

### Task / Goal Input and VLM Planner

Current owner: `episode.run_episode` constructs `PlannerContext`; `SequentialPlanner` owns fixed-order progression.

Gaps:

- no VLM planner service or API;
- no scene image/observation payload in `PlannerContext`;
- no planner deadline, timeout, retry policy, or model provenance;
- planner silently advances to the next skill after retries exceed `max_retries`, even after failure (`planner.py:93-103`), which must become an explicit `REPLAN/FAILED` policy;
- `tracking/inference.json` says `VLM Planner` and predicate verifier, but is not the runtime registry and contradicts the obs-only current path. It cannot be presented as current deployed truth.

### Skill Contract Registry and SkillCall

Current owner: `skills.SkillRegistry` + `bindings.CONTRACTS`.

Strengths:

- argument validation and exact GRASP/MOVE/PLACE instructions/budgets exist;
- max steps are GRASP 208, MOVE 288, PLACE 96;
- tests reject missing args.

Gaps:

- canonical data is duplicated between executable Python bindings and `tracking/inference.json`;
- no schema version/contract revision in calls or traces;
- `SkillCall` lacks instruction, budget, contract revision, call id, and retry policy;
- obs-only manager bypasses executable `can_start` as privileged and records `can_start=True` unconditionally (`executor.py:80-82`); a real robot-safe obs-derived precondition contract is unresolved.

### Execution Manager / Router

Current owner: `executor.ExecutionManager`.

Strengths:

- manager owns step budgets and local transitions;
- policy chunk is fetched before action execution;
- a stale result cannot currently cross a skill boundary only because verifier RPC is synchronous and manager waits for it.

Gaps:

- synchronous verifier work pauses the control loop;
- no 20 Hz deadline scheduler or overrun handling;
- no async queue, cancellation, backpressure, in-flight cap, response identity, or stale guard;
- `RETRY` exists in the enum but verifier never emits it; retry/replan semantics are split across string hints and scripted planner behavior;
- episode budget is checked only at planner boundaries, so a final skill can overrun the remaining global budget;
- `SuccessGate.observe` suppresses all exceptions (`executor.py:116-119`), hiding endpoint failures from telemetry.

### Async Verifier Client

Current owner: none. `RemoteVLMScorerBackend` is a synchronous socket client.

Missing acceptance-critical behavior:

- queue size 1 / latest replacement;
- max one in flight;
- episode/skill/step/request identity;
- stale response rejection;
- connect/read/write timeout;
- reconnect and bounded retry;
- request cancellation/drop accounting;
- asynchronous callback/future into a local transition state machine.

Also, current event gating can issue a request on every subsequent settled step because `_should_query` allows `candidate_stop` whenever one step has elapsed (`obs_verifier.py:324-328`). That is not the target “one rising-edge candidate-stop request.”

### Shared GPU Inference Server

Current owner: `infer_verify_server.InferVerifyServer`.

Strengths:

- one process and one model object;
- policy and VLM logical operations share the same GPU/model;
- lock guarantees no concurrent CUDA forward in this process.

Gaps:

- no separate policy/verifier queues;
- no policy priority; lock acquisition is race/FIFO-undefined;
- no latest-only verifier drop before start;
- no request metadata echo, queue wait, model latency, timeout, health/version endpoint, maximum request size, or structured error response;
- pickle over a raw socket is trusted-network-only and has no authentication/integrity boundary;
- connection threads are unbounded and each can wait on the lock;
- current canonical `xiaomi-cu121/run.sh server` launches stock `upstream/deploy/server.py`, not `infer_verify_server.py` (`xiaomi-cu121/run.sh:34-38`). The combined server is therefore an experimental/manual path, not canonical deployment.

### RoboCasa / Robot Control

Current owner: `environment.RoboCasaEnvironment` and `xiaomi-cu121/rollout.py`.

Strengths:

- camera and proprio construction matches policy inputs;
- one decoded action is passed to each `env.step`;
- 16-step action chunk boundary is enforced by policy client.

Gaps:

- simulator loop is not wall-clock paced at 20 Hz;
- no action deadline/watchdog or safe-stop contract;
- observation timestamps and episode/step IDs are not carried into verifier requests;
- current harness reports `video_fps=20`, which is encoding metadata, not proof of 20 Hz control timing.

### Frozen verifier and Endpoint SuccessGate

Current owners: `obs_verifier.py`, `success_gate.py`, `vlm_backends.py`.

Strengths:

- obs-only schema guard exists;
- local hysteresis/sequence transition decision exists;
- strict gate is logically separate from boundary latch;
- measured P(yes) latency and known accuracy limitations are preserved.

Gaps:

- boundary and endpoint calls are not distinguishable in server requests;
- request view is applied client-side when composing pixels but not sent/echoed as metadata;
- SuccessGate samples recorded frames during execution rather than issuing one explicit final frame/window request;
- strict result does not control planner transition; boundary `ADVANCE` can produce `SkillStatus.SUCCESS` while strict gate fails;
- MOVE has no strict gate;
- verifier quality remains insufficient for a deployment claim. Existing three-seed published evidence reports official success 0/3 and obs/sim agreement 2/3; later sequence artifacts are experiment evidence, not an async-deployment validation.

### Trace / Telemetry

Current owner: `trace.Trace` plus summary JSON.

Strengths:

- plan, route, action window, frames, VLM score, transition and final result are persisted as JSONL;
- per-skill aggregate VLM time/mean latency is recorded.

Gaps:

- no request/response correlation key;
- no queue wait, client round-trip, timeout, retry, drop, stale rejection, scheduler decision, policy latency, or control deadline miss;
- no explicit checkpoint hash/server build/model revision in every request/result trace;
- trace uses local wall-clock seconds without monotonic duration fields.

### Offline GT

Current owner: `episode.py` and generated evaluation artifacts.

Status: runtime isolation is present for the `obs_only=True` branch, but enforcement is branch-based rather than an architectural dependency boundary. `SkillResult` is built with `env.predicates()` even in the VLM path (`executor.py:128-167`); `as_dict()` currently omits those predicates, which prevents trace leakage by serialization accident rather than by type. Replace this with an explicit offline-evaluation envelope before production.

## 5. Timeout, retry, stale-data, and failure semantics

| Concern | Current | Required before target deployment |
|---|---|---|
| Skill timeout | per-contract step budget returns TIMEOUT/REPLAN | retain; also honor remaining episode budget |
| Planner retry | fixed planner retries once, then skips failed skill | explicit retryability and terminal/replan policy; never silently skip |
| Policy RPC timeout | none | deadline shorter than chunk depletion; bounded reconnect/fail-safe |
| Verifier RPC timeout | none | bounded timeout; timeout means local CONTINUE or policy-defined conservative action, never blind advance |
| Queue backlog | each caller blocks; connection threads may accumulate | verifier pending capacity 1, replace newest, one in flight |
| Stale response | impossible only because sync | compare episode_id + skill_id + observation_step/request_id before state mutation |
| Endpoint failure | exceptions swallowed by SuccessGate caller | structured failure telemetry and strict FAIL/UNKNOWN |
| Environment done/trunc | maps to FAILED | retain with final observation and reason |
| Server crash | socket exception propagates | manager safe-state, reconnect budget, trace, no skill transition |

## 6. Checkpoint, deploy, and rollback boundaries

### Current

- Base checkpoint is mounted read-only at `/checkpoint` and per-file hashes exist in `xiaomi-cu121/checkpoint.sha256`.
- `BasePolicyClient.provenance()` records only model path, adapter mode, adapter checkpoint, and replan steps; it does not record checkpoint hash/revision.
- `xiaomi-cu121/run.sh server` starts the stock policy server. There is no canonical command for the combined infer+verify server.
- `scripts/deploy.sh` deploys only `rl-train-t_3ed65912` and `rl-env-t_4f3f2b20`; it does not deploy `architecture_smoke` (`scripts/deploy.sh:35-45`).
- Existing rollback is manual: stop an experimental server and restart the stock policy server. No atomic alias/image tag switch, health verification, or scripted rollback record exists.

### Required deployment gate

1. Pin source commit, container digest, checkpoint revision and full checkpoint manifest.
2. Build a production server entrypoint containing both logical APIs and explicit priority scheduler.
3. Add `/health` or equivalent RPC returning server build, checkpoint/model revision, queue depths and readiness.
4. Run stock-policy parity on fixed inputs before enabling verifier traffic.
5. Run scheduler tests proving policy priority, queue-size-one replacement, max-one verifier in flight, stale rejection and bounded timeout.
6. Run timed 20 Hz integration with policy+verifier contention; publish p50/p95/p99 policy latency, verifier queue/model/RTT, and deadline misses.
7. Canary with verifier transitions in shadow mode first; compare against current manager decisions without controlling the robot.
8. Promote by immutable image/digest and config revision.
9. Rollback atomically to the pinned stock policy server/image and previous manager config; verify health and a fixed inference probe. Do not mutate the checkpoint in place.

## 7. Readiness gates for UI implementation approval

The module-level UI diagram can be built after the wording below is accepted, but it must not imply deployment.

- Label entire shared-server enclosure: `Proposed deployment topology — SAME physical GPU · ONE shared Xiaomi/Qwen3-VL model instance`.
- Inside it show `Policy API / high-priority queue`, `Verifier API / low-priority latest-only queue`, and `priority scheduler / serialized CUDA` as target components.
- Place `ADVANCE / CONTINUE / RETRY / REPLAN` inside control-host Manager/Verifier Client, not inside the GPU server.
- Label current planner separately: `Current: SequentialPlanner scripted obs-only stub — NOT learned VLM planner`.
- Use a distinct current-status table; do not color target modules as deployed/healthy.
- Keep GT in a separate evaluation-only plane with no runtime arrow.
- Do not say separate GPU, remote GPU, zero overhead, or every-step verifier.
- Preserve exact GRASP/MOVE/PLACE contract cards.
- Put verifier accuracy evidence and known limitations below contracts.

## 8. Verification performed for this audit

- Canonical modules read and cross-checked: planner, schemas, bindings/registry, manager, policy client, environment, verifier, SuccessGate, backend RPC, shared server, episode loop, trace, deployment scripts and checkpoint manifests.
- Measured artifacts checked: sequence summaries for seeds 40/41/42 and optimized summary latency samples.
- No live page/UI file was changed or deployed.
- This report is the only task-owned source change.
