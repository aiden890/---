"""Strict success judge -- the SECOND verifier role (task t_32a4f9b6).

The obs-only pipeline has TWO distinct judgment roles that the original design
conflated under one tau/latch:

  1. BOUNDARY judgment (``obs_verifier.ObsVLMVerifier``): an ONLINE, per-frame
     latch that decides when to ADVANCE/CONTINUE/REPLAN a skill during the
     rollout. It may be fast and lenient -- advancing a skill a little early is
     cheap and recoverable.

  2. SUCCESS judgment (THIS module, ``SuccessGate``): a STRICT, calibrated
     judge of whether the skill/task ACTUALLY succeeded, used for the episode
     report and as the gate a downstream reward would trust. It must not be
     fooled by the "놓는 순간 flush" false positive: a single frame where the lid
     momentarily looks seated while the gripper is still there.

Why they must differ (measured, see CALIBRATION_LOG.md): on the balanced
30-episode/skill corpus, the online transition latch cannot reach precision 0.9
at useful recall. A separately calibrated completion checkpoint can do so for
PLACE and MOVE because it judges the final observation/window.

  * VIEW ROUTING: score a single camera view, not the diluted concat.
  * ENDPOINT / TEMPORAL AGGREGATION: PLACE uses the final eye-in-hand frame;
    GRASP uses a final 10-frame mean. These choices were selected from the same
    question x view x aggregation sweep and are not shared with the online latch.

The per-skill (view, question, aggregation, threshold) below are the operating
points the offline sweep selected against the sim GT label. The sim GT is used
ONLY to CALIBRATE these constants offline; ``SuccessGate`` at runtime reads
obs-derived P(yes) only (same obs-only invariant as the boundary judge).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional

from obs_verifier import VLMBackend

# Success-judge questions (single yes/no visual facts; no privileged pose). The
# retreat/clear fact is the most discriminative for PLACE (see calibration).
SUCCESS_QUESTIONS = {
    "GRASP_OBJECT": (
        "These are camera views of a robot manipulation scene. Is the robot "
        "gripper firmly grasping and holding the blender lid, having lifted it "
        "clear off the counter? Answer yes or no."
    ),
    "PLACE_OBJECT_combined": (
        "These are camera views of a robot manipulation scene showing a "
        "blender. Has the blender lid been placed back on top of the blender "
        "base so the blender is closed, AND has the robot gripper let go of the "
        "lid and moved away from it? Answer yes or no."
    ),
    "PLACE_OBJECT_seated": (
        "These are camera views of a robot manipulation scene showing a "
        "blender. Is the blender lid resting flush and fully seated on top of "
        "the blender base, so the blender is properly closed? Answer yes or no."
    ),
    "PLACE_OBJECT_clear": (
        "These are camera views of a robot manipulation scene. Is the robot "
        "gripper empty and moved away, no longer touching or holding the "
        "blender lid? Answer yes or no."
    ),
}


@dataclass(frozen=True)
class SubQuestion:
    """One sub-fact of a success judgment: (question, camera view, threshold).

    ``threshold`` is this sub-fact's own calibrated cut on the aggregated
    per-frame P(yes). A skill succeeds only when EVERY sub-fact clears its own
    threshold (AND), so a primary fact and a weaker guard fact can use different
    cuts instead of being forced onto one shared min-threshold.
    """
    question: str
    view: str = "full"          # 'full'|'left'|'right'|'eye'
    threshold: float = 0.75


@dataclass(frozen=True)
class SuccessCriterion:
    """Calibrated strict success judge for one skill.

    Each ``subs`` fact is aggregated over the observation window by ``aggregate``
    ('mean' over the executed segment, or 'last_k_mean') and compared to its OWN
    ``SubQuestion.threshold``; the skill succeeds only if ALL sub-facts pass
    (AND). ``min_frames`` guards against declaring success from too short a
    window (temporal stability).
    """
    skill: str
    subs: List[SubQuestion]
    aggregate: str = "mean"     # 'mean' | 'last_k_mean'
    last_k: int = 5
    min_frames: int = 3
    note: str = ""


# ---- calibrated operating points (offline sweep, balanced 15+/15- per skill) -- #
# PLACE: final combined@eye >= 0.8997285 -> precision .90 / recall .60.
# GRASP: final-10 mean grasp@eye >= 0.4532653 is the best-F1 honest point ->
#        precision .667 / recall .933. Precision .90 is unattainable at useful
#        recall (the only >=.90 points recall <=.133), so do not claim otherwise.
SUCCESS_CRITERIA = {
    "PLACE_OBJECT": SuccessCriterion(
        skill="PLACE_OBJECT",
        subs=[
            SubQuestion(SUCCESS_QUESTIONS["PLACE_OBJECT_combined"],
                        view="eye", threshold=0.899728536605835),
        ],
        aggregate="last_k_mean",
        last_k=1,
        min_frames=5,
        note="balanced n=30: final combined@eye>=0.89973; precision .90 recall .60",
    ),
    "GRASP_OBJECT": SuccessCriterion(
        skill="GRASP_OBJECT",
        subs=[SubQuestion(SUCCESS_QUESTIONS["GRASP_OBJECT"],
                          view="eye", threshold=0.453265318274498)],
        aggregate="last_k_mean",
        last_k=10,
        min_frames=5,
        note="balanced n=30 best-F1: final10 grasp@eye>=0.45327; precision .667 recall .933",
    ),
    # MOVE has no strict success gate (it is a waypoint, not a task outcome).
}


@dataclass
class SuccessGate:
    """Accumulates a skill's executed frames, then renders a strict verdict.

    Runtime-obs only: it stores per-frame P(yes) for each calibrated sub-fact
    (scored via the SAME VLM backend as the boundary judge, view-routed) and,
    when the skill segment ends, reduces them to a pass/fail. No sim predicate
    is ever read here.
    """
    backend: VLMBackend
    criterion: SuccessCriterion
    _per_sub: List[List[float]] = field(default_factory=list)
    n_frames: int = 0

    def __post_init__(self):
        self._per_sub = [[] for _ in self.criterion.subs]

    def observe(self, images: Mapping[str, Any]) -> None:
        """Score every calibrated sub-fact on this frame (view-routed)."""
        for i, sub in enumerate(self.criterion.subs):
            p = float(self.backend.score_view(images, sub.question, view=sub.view))
            self._per_sub[i].append(p)
        self.n_frames += 1

    def _agg(self, vals: List[float]) -> Optional[float]:
        if not vals:
            return None
        if self.criterion.aggregate == "last_k_mean":
            tail = vals[-self.criterion.last_k:]
            return sum(tail) / len(tail)
        return sum(vals) / len(vals)  # 'mean'

    def verdict(self) -> dict:
        """Strict success verdict from the accumulated window (obs-only).

        Each calibrated sub-fact is aggregated over the window and compared to
        its OWN threshold; success requires ALL sub-facts to pass (AND).
        """
        c = self.criterion
        sub_aggs = [self._agg(v) for v in self._per_sub]
        sub_thr = [s.threshold for s in c.subs]
        sub_pass = [(a is not None and a >= t) for a, t in zip(sub_aggs, sub_thr)]
        if self.n_frames < c.min_frames:
            return {"success": False, "reason": "insufficient_window",
                    "n_frames": self.n_frames,
                    "sub_scores": sub_aggs, "sub_thresholds": sub_thr,
                    "sub_views": [s.view for s in c.subs], "criterion": c.note}
        ok = bool(sub_aggs) and all(sub_pass)
        return {
            "success": bool(ok),
            "reason": "strict_success_gate" if ok else "below_threshold",
            "n_frames": self.n_frames,
            "sub_scores": sub_aggs,
            "sub_thresholds": sub_thr,
            "sub_pass": sub_pass,
            "sub_views": [s.view for s in c.subs],
            "aggregate": c.aggregate,
            "criterion": c.note,
        }


def make_success_gate(skill: str, backend: VLMBackend) -> Optional[SuccessGate]:
    """Return a SuccessGate for a skill, or None if the skill has no strict gate."""
    crit = SUCCESS_CRITERIA.get(skill)
    if crit is None:
        return None
    return SuccessGate(backend=backend, criterion=crit)
