"""VLM planner adapter.

IMPORTANT: this smoke test ships an ORACLE / SCRIPTED planner stub, NOT a
learned VLM planner. No skill-conditioned VLM planner has been trained yet;
the architecture contract only requires that the planner *interface* is honored
(input PlannerContext -> output a schema-valid SkillCall, no policy/sim access).
The stub selects the next skill from the current predicate state using the same
subgoal ordering a competent planner would produce for CloseBlenderLid. Never
report its output as a learned planner.

The class only depends on schemas + the SkillRegistry catalog; it makes no
simulator or policy calls and holds no environment handle.
"""
from __future__ import annotations

from schemas import PlannerContext, SkillCall, SkillResult, SkillStatus
from skills import SkillRegistry


class OraclePlanner:
    """Scripted planner stub honoring the planner interface.

    Policy (CloseBlenderLid subgoal chain):
      not grasped              -> GRASP_OBJECT
      grasped, not in preplace -> MOVE_OBJECT
      grasped, in preplace     -> PLACE_OBJECT
      task predicate satisfied -> None (episode done)

    On a FAILED/TIMEOUT last result the planner re-issues the same subgoal
    (retry/replan) rather than skipping ahead -- demonstrating the planner can
    advance OR retry without manual intervention.
    """

    kind = "oracle_stub"  # provenance tag written into every trace record

    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def plan(self, ctx: PlannerContext) -> SkillCall | None:
        p = ctx.predicates
        if p.get("official_check_success"):
            return None  # task complete -> terminal

        grasped = bool(p.get("lid_grasped"))
        in_preplace = bool(p.get("in_preplace_region"))

        if not grasped:
            name = "GRASP_OBJECT"
        elif not in_preplace:
            name = "MOVE_OBJECT"
        else:
            name = "PLACE_OBJECT"

        contract = self.registry.get(name)
        return SkillCall(name=name, args=dict(contract.default_args))

    @staticmethod
    def rationale(call: SkillCall | None, last: SkillResult | None) -> str:
        if call is None:
            return "task predicate satisfied; terminate episode"
        base = f"selected {call.name} from current predicate state"
        if last is not None and last.status is not SkillStatus.SUCCESS:
            return f"{base} (retry after previous {last.skill}={last.status.value})"
        return base


class SequentialPlanner:
    """Obs-only planner: advances GRASP_OBJECT -> MOVE_OBJECT -> PLACE_OBJECT on the
    verifier's SUCCESS, NOT on any privileged simulator predicate.

    Used with the obs-only VLM verifier so the ENTIRE runtime (skill termination
    AND next-skill selection) reads only what the robot receives: the VLM verifier
    ADVANCE (obs P(yes) latch) drives progression; a FAILED/TIMEOUT re-issues the
    same skill up to ``max_retries`` before moving on. It never queries
    ``ctx.predicates`` (the sim ground-truth), so the pipeline is transferable.

    This is still a SCRIPTED stub (fixed subgoal order), NOT a learned planner.
    """

    kind = "sequential_obs_stub"
    SEQUENCE = ("GRASP_OBJECT", "MOVE_OBJECT", "PLACE_OBJECT")

    def __init__(self, registry: SkillRegistry, max_retries: int = 1):
        self.registry = registry
        self.max_retries = max_retries
        self._idx = 0
        self._retries = 0
        self._last_seen = None

    def plan(self, ctx: PlannerContext) -> SkillCall | None:
        last = ctx.last_result
        # advance/retry based ONLY on the obs-only verifier outcome
        if last is not None and last is not self._last_seen:
            self._last_seen = last
            if last.status is SkillStatus.SUCCESS:
                self._idx += 1
                self._retries = 0
            else:
                self._retries += 1
                if self._retries > self.max_retries:
                    self._idx += 1        # give up on this skill, move on honestly
                    self._retries = 0
        if self._idx >= len(self.SEQUENCE):
            return None                    # all skills attempted -> episode done
        name = self.SEQUENCE[self._idx]
        contract = self.registry.get(name)
        return SkillCall(name=name, args=dict(contract.default_args))

    @staticmethod
    def rationale(call: SkillCall | None, last: SkillResult | None) -> str:
        if call is None:
            return "all skills attempted (obs-only verifier); terminate episode"
        base = f"sequential obs-only plan -> {call.name}"
        if last is not None and last.status is not SkillStatus.SUCCESS:
            return f"{base} (retry after {last.skill}={last.status.value})"
        return base
