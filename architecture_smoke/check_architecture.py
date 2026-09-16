"""Offline architecture checks -- run on any CPU box (no GPU/numpy/sim needed).

Covers the five required focused checks before GPU runs:
  1. schema validation        (SkillCall arg schema enforced)
  2. instruction rendering     (fixed NL string from template + args)
  3. adapter-disabled assertion (policy refuses any adapter checkpoint)
  4. verifier handoff          (CONTINUE -> ADVANCE / REPLAN transitions)
  5. mocked end-to-end loop     (planner->skill->verify->advance->planner, task success)

Run: python3 check_architecture.py   (exit 0 = all pass)
"""
from __future__ import annotations

import sys

from schemas import (AdapterMode, Decision, PlannerContext, PolicyInput, SkillCall,
                     SkillStatus)
import bindings
from skills import SkillRegistry
from planner import OraclePlanner
from policy import MockPolicy
from environment import MockEnvironment
from executor import ExecutionManager
from verifier import PredicateVerifier
from trace import Trace
from episode import run_episode

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{PASS if cond else FAIL}] {name}" + (f" -- {detail}" if detail else ""))


def main():
    reg = SkillRegistry(bindings.CONTRACTS)

    # 1. schema validation ---------------------------------------------------
    ok = False
    try:
        reg.validate_call(SkillCall("GRASP_OBJECT", {"object": "blender_lid"}))  # missing grasp_region
    except ValueError:
        ok = True
    check("schema_validation.missing_arg_rejected", ok)
    reg.validate_call(SkillCall("GRASP_OBJECT", {"object": "blender_lid", "grasp_region": "lid_handle"}))
    check("schema_validation.valid_call_accepted", True)
    ok = False
    try:
        reg.get("NOT_A_SKILL")
    except KeyError:
        ok = True
    check("schema_validation.unknown_skill_rejected", ok)

    # 2. instruction rendering ----------------------------------------------
    rendered = reg.render(SkillCall("PLACE_OBJECT", {"object": "blender_lid", "destination": "blender"}))
    check("instruction_rendering.place", rendered == bindings.PLACE_INSTRUCTION, rendered)
    rendered_g = reg.render(SkillCall("GRASP_OBJECT", dict(reg.get("GRASP_OBJECT").default_args)))
    check("instruction_rendering.grasp", rendered_g == bindings.GRASP_INSTRUCTION, rendered_g)

    # 3. adapter-disabled assertion -----------------------------------------
    mock = MockPolicy(replan_steps=4)
    prov = mock.provenance()
    check("adapter_disabled.provenance_none", prov["adapter_checkpoint"] is None and prov["adapter_mode"] == "disabled")
    ok = False
    try:
        mock.infer(PolicyInput(instruction="x", state_history=[], image_history={},
                               adapter_mode=AdapterMode.LORA, adapter_checkpoint="grasp.lora"))
    except AssertionError:
        ok = True
    check("adapter_disabled.lora_checkpoint_refused", ok)
    out = mock.infer(PolicyInput(instruction="x", state_history=[], image_history={},
                                 adapter_mode=AdapterMode.DISABLED))
    check("adapter_disabled.base_infer_ok", out.adapter_checkpoint is None and out.chunk_len == 4)

    # 4. verifier handoff ----------------------------------------------------
    move = reg.get("MOVE_OBJECT")
    v = PredicateVerifier(move)
    # not grasped/in-preplace -> CONTINUE
    r1 = v.update({"lid_grasped": False, "in_preplace_region": False})
    check("verifier.continue_when_unmet", r1.decision is Decision.CONTINUE)
    # grasped + in preplace for hold_steps (3) -> ADVANCE on 3rd
    d = None
    for _ in range(move.hold_steps):
        d = v.update({"lid_grasped": True, "in_preplace_region": True})
    check("verifier.advance_after_hold", d.decision is Decision.ADVANCE, d.reason)
    # timeout -> REPLAN
    g = reg.get("GRASP_OBJECT")
    v2 = PredicateVerifier(g)
    dd = None
    for _ in range(g.max_steps):
        dd = v2.update({"lid_grasped": False})
    check("verifier.replan_on_timeout", dd.decision is Decision.REPLAN, dd.reason)

    # 5. mocked end-to-end loop ---------------------------------------------
    import tempfile, os
    tmp = tempfile.mkdtemp(prefix="arch_smoke_check_")
    env = MockEnvironment(seed=0, replan_steps=4, grasp_after=6, move_after=4, place_after=3)
    planner = OraclePlanner(reg)
    trace = Trace(os.path.join(tmp, "trace.jsonl"), {"seed": 0, "check": True})
    manager = ExecutionManager(reg, MockPolicy(replan_steps=4), env, trace, AdapterMode.DISABLED)
    summary = run_episode(planner, manager, env, trace, reg, bindings.GOAL, episode_budget=400)
    trace.close()
    skills_seq = [s["skill"] for s in summary["skills"]]
    statuses = [s["status"] for s in summary["skills"]]
    check("e2e.reaches_task_success", summary["task_success"] and summary["terminal"] == "task_success",
          f"skills={skills_seq} statuses={statuses}")
    check("e2e.full_skill_chain", skills_seq == ["GRASP_OBJECT", "MOVE_OBJECT", "PLACE_OBJECT"],
          str(skills_seq))
    check("e2e.all_skills_succeeded", all(s == SkillStatus.SUCCESS.value for s in statuses), str(statuses))
    # trace has plan + skill_result + episode_end records
    import json
    recs = [json.loads(l) for l in open(os.path.join(tmp, "trace.jsonl"))]
    kinds = {r["type"] for r in recs}
    check("e2e.trace_has_boundaries", {"plan", "route", "verify", "skill_result", "episode_end"} <= kinds,
          str(sorted(kinds)))

    # ---- verdict -----------------------------------------------------------
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results)-n_fail}/{len(results)} checks passed.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
