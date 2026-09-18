# Skill VLA asynchronous control-plane interface

`ExecutionManager` uses the latest-only asynchronous path by default whenever a
VLM backend is configured. `--sync-verifier` is the single compatibility switch
in `run_vlm_inference.py`; it restores the previous blocking path.

## Canonical envelope

`async_control.ControlRequest` and `ControlResponse` carry the same identity:

- `episode_id`
- `skill_id`
- `observation_step`
- `request_id`
- `request_kind` (`boundary`, `endpoint`, or `planner`)

Runtime payloads are observation-only. Recursive schema validation rejects
simulator predicates, official success, ground-truth labels, and privileged
fields. A response is advisory data. Only `ExecutionManager` validates its
`recommendation` and applies `ADVANCE`, `CONTINUE`, `RETRY`, or `REPLAN`.

## Queue and failure contract

- Exactly one RPC may be in flight.
- There is one not-yet-started pending slot. Submitting newer work atomically
  replaces that slot; an in-flight GPU forward is never preempted.
- Episode/skill mismatches and observations older than the last accepted step are
  dropped with `stale_episode`, `stale_skill`, or `stale_step` telemetry.
- A timeout may retry once with the identical `request_id` and identity. The
  transport must honour the supplied deadline; a timed-out framed socket is
  reconnected before retry.
- Shutdown either cancels pending work or drains it, then joins the worker. A
  transport violating its deadline causes an explicit shutdown `TimeoutError`.
- Boundary responses can only nominate a candidate stop. The manager schedules
  the strict `endpoint` SuccessGate flow and advances only after that result.

Queue submit/supersede, RPC start/done, timeout, retry, stale/drop, cancel, error,
and shutdown events are written as `async_control` trace records without image
or proprio payloads.

## Shared-GPU server scheduling

`infer_verify_server.py` loads one Xiaomi/Qwen3-VL model object and exposes two
logical lanes through `PriorityDispatcher`:

- requests without an `op` retain the stock policy wire API and enter the FIFO
  high-priority policy queue;
- `op=background_vlm` requires the canonical envelope above and enters one
  latest-only background slot (`boundary`, `endpoint`, or `planner`).

One dispatcher thread performs all CUDA forwards. A running background forward
is non-preemptive, but once it completes all waiting policy work is selected
before pending background work. Replacing an unstarted background request returns
`dropped`; expired work returns `timeout`. `op=health` reports readiness, queue
depths, drop/supersede/stale/timeout counters, per-kind queue wait and model
latency, and policy delay caused by a running background forward.

## Observation-only planner

`run_vlm_inference.py` now has one planner selector: `--planner-mode vlm` is the
target and default path; `--planner-mode sequential` preserves deterministic
legacy-rollout behavior. The VLM path submits `request_kind=planner` through the
same background slot and the same model instance. It never creates a planner
process or a second GPU model.

The planner receives only goal text, the current three-camera observation,
proprio14, and plan/result history. Result history deliberately excludes the
simulator predicate dictionary. Its output must be exactly the five-field JSON
object `SkillCall{name,args,instruction,contract,budget}`. Name, exact argument
keys, rendered instruction, done contract, and positive bounded budget are all
checked against `tracking/inference.json`'s bindings in `bindings.CONTRACTS`.
Unknown skills, missing or extra arguments, malformed JSON, stale identities,
timeouts, model errors, over-budget calls, repeated cycles, and plan-length
overflow cannot reach robot control. They either use the bounded deterministic
`SequentialPlanner` fallback (with the reason in the plan trace) or terminate
safely at the maximum plan length. Planner quality remains unclaimed until the
separate GPU evaluation gate; the local smoke establishes wiring and safety only.
