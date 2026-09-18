"""Outer control loop: Planner -> SkillCall -> Manager -> base VLA -> env ->
Verifier -> SkillResult -> Planner, until the official task predicate succeeds
or the episode budget is exhausted.

This is generic orchestration -- it holds no task predicate logic and no policy
internals; it only wires the planner and the ExecutionManager together and
records planner boundaries into the trace.
"""
from __future__ import annotations

from schemas import PlannerContext, SkillStatus


def run_preplanned_episode(planner, manager, env, trace, registry, goal, episode_budget):
    """Plan once from the reset snapshot, then consume the canonical plan.

    Planning and strict validation finish before ``manager.execute`` can issue an
    action. Any planner, snapshot, or budget error returns with zero actions and
    no scripted replacement. Skill results never trigger another planner call.
    """
    from pre_episode_planner import PlanRejected

    env.reset()
    reset_observation = env.obs_for_verifier()
    context = PlannerContext(
        goal=goal, predicates={}, skill_catalog=registry.names,
        step_budget_remaining=episode_budget, planner_calls=0,
        observation=reset_observation,
    )
    try:
        plan = planner.plan_episode(context)
        # Re-read immediately before handing the first call to the executor. The
        # simulator must still be at the exact reset snapshot while planning.
        planner.validate_execution_snapshot(plan, env.obs_for_verifier())
        if sum(call.budget or 0 for call in plan.calls) > episode_budget:
            raise PlanRejected("canonical full plan exceeds episode budget")
    except PlanRejected as exc:
        summary = {
            "seed": env.seed, "goal": goal, "terminal": "plan_rejected",
            "planner_calls": planner.calls, "runtime_boundary_planner_calls": 0,
            "steps_used": 0, "task_success": False, "obs_task_success": None,
            "skills": [], "final_predicates": {}, "plan_error": str(exc),
        }
        trace.episode_end(terminal=summary["terminal"], planner_calls=planner.calls,
                          steps_used=0, task_success=False)
        return summary

    steps_used = 0
    skill_log = []
    terminal = "plan_consumed"
    for boundary_index, call in enumerate(plan.calls):
        trace.plan(
            boundary_index, env.observation_ref(), {}, call.as_dict(),
            registry.render(call), "canonical pre-episode plan", planner.kind)
        result = manager.execute(call)
        steps_used += result.steps
        skill_log.append(result.as_dict())
        trace.skill_result(result.as_dict(), next_skill=(
            "advance_preplanned" if boundary_index + 1 < len(plan.calls) else None))

    final_pred = env.predicates()
    obs_task_success = None
    for result in reversed(skill_log):
        if result.get("skill") == "PLACE_OBJECT" and result.get("success_gate") is not None:
            obs_task_success = bool(result["success_gate"].get("success"))
            break
    summary = {
        "seed": env.seed, "goal": goal, "terminal": terminal,
        "planner_calls": planner.calls, "runtime_boundary_planner_calls": 0,
        "steps_used": steps_used,
        "task_success": bool(final_pred.get("official_check_success")),
        "obs_task_success": obs_task_success, "skills": skill_log,
        "final_predicates": final_pred,
    }
    trace.episode_end(terminal=terminal, planner_calls=planner.calls,
                      steps_used=steps_used, task_success=summary["task_success"])
    return summary


def run_episode(planner, manager, env, trace, registry, goal, episode_budget,
                max_planner_calls=12, obs_only=False):
    """Drive one end-to-end episode. Returns a machine-readable summary dict.

    ``obs_only=True`` (VLM-verifier pipeline): skill termination and next-skill
    selection are obs-only, so the loop does NOT break on the privileged sim
    ``official_check_success`` predicate mid-episode -- it runs until the planner
    returns None (all skills attempted) or the budget is exhausted. The sim
    predicate is still read ONCE at the end and recorded as the offline
    ``task_success`` evaluation label (never a runtime judge input).
    """
    env.reset()
    planner_calls = 0
    steps_used = 0
    last_result = None
    skill_log = []
    plan_history = []
    terminal = "budget_exhausted"

    while planner_calls < max_planner_calls and steps_used < episode_budget:
        pred = {} if obs_only else env.predicates()
        planner_observation = env.obs_for_verifier() if obs_only else None
        ctx = PlannerContext(goal=goal, predicates=pred, skill_catalog=registry.names,
                             last_result=last_result,
                             step_budget_remaining=episode_budget - steps_used,
                             planner_calls=planner_calls,
                             observation=planner_observation,
                             plan_history=tuple(plan_history))
        call = planner.plan(ctx)
        rationale = planner.rationale(call, last_result)
        trace.plan(planner_calls, env.observation_ref(), (env.predicates() if not obs_only else {}),
                   call.as_dict() if call else None,
                   registry.render(call) if call else None,
                   rationale, planner.kind)
        planner_calls += 1

        if call is None:
            terminal = "planner_done" if obs_only else "task_success"
            break

        plan_history.append({"call": call.as_dict()})
        result = manager.execute(call)
        steps_used += result.steps
        last_result = result
        skill_log.append(result.as_dict())
        plan_history[-1]["result"] = {
            "status": result.status.value, "skill": result.skill,
            "steps": result.steps, "terminated_by": result.terminated_by,
            "reason": result.reason,
        }

        # planner sees the result and decides next on the next loop turn.
        next_hint = None
        if result.status is SkillStatus.SUCCESS:
            next_hint = "advance"
        elif result.status is SkillStatus.TIMEOUT:
            next_hint = "replan"
        else:
            next_hint = "retry"
        trace.skill_result(result.as_dict(), next_skill=next_hint)

        if not obs_only and env.predicates().get("official_check_success"):
            terminal = "task_success"
            break

    final_pred = env.predicates()
    # obs-only task success: the strict PLACE success gate's verdict (view-routed,
    # episode-level). This is the runtime-usable success signal (NO sim predicate),
    # reported ALONGSIDE the offline sim ``task_success`` label so the reader can
    # see boundary-advance vs strict-success vs sim-GT for each episode.
    obs_task_success = None
    for r in reversed(skill_log):
        if r.get("skill") == "PLACE_OBJECT" and r.get("success_gate") is not None:
            obs_task_success = bool(r["success_gate"].get("success"))
            break
    summary = {
        "seed": env.seed, "goal": goal, "terminal": terminal,
        "planner_calls": planner_calls, "steps_used": steps_used,
        "task_success": bool(final_pred.get("official_check_success")),
        "obs_task_success": obs_task_success,
        "skills": skill_log,
        "final_predicates": final_pred,
    }
    trace.episode_end(**{k: summary[k] for k in
                         ("terminal", "planner_calls", "steps_used", "task_success")})
    return summary
