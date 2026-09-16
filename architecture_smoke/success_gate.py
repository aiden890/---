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

Why they must differ (measured, see CALIBRATION_LOG.md): on the calibration
corpus the single-frame boundary signal on the wide 3-cam concat CANNOT separate
PLACE success from failure -- failures reach P(yes) as high as successes, so any
tau/latch/hold on that signal yields precision ~0.22. Two calibrated changes fix
the SUCCESS judge to precision 1.0 / recall 1.0 on the corpus:

  * VIEW ROUTING: score a single camera view, not the diluted concat. PLACE's
    "gripper released and clear" fact is cleanest from the right agentview
    (place_clear@right rollout-AUC 1.0 vs 0.31 for the concat); GRASP's hold is
    cleanest from the eye-in-hand view.
  * EPISODE-LEVEL TEMPORAL AGGREGATION: the strict judge summarises P(yes) over
    a window of the skill's final frames (mean over the whole executed segment
    for the corpus operating point) rather than trusting one frame. This is the
    "release 후 N프레임 관찰 -> 계속 seated + 그리퍼 후퇴 유지" requirement made
    concrete: a transient flush at the release instant cannot raise the window
    mean above threshold, but a genuinely-closed-and-cleared end state can.

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


# ---- calibrated operating points (offline sweep vs sim GT, corpus g10/p10) -- #
# PLACE: clear@right episode-mean >= 0.75 is the clean separator (pos [0.78,0.97]
#        vs neg max 0.73 -> AUC 1.0). seated@right episode-mean >= 0.50 is ANDed
#        as a guard (pos >=0.52) so a cleared-but-not-seated end state (lid on
#        the floor / knocked off) also fails. Together: precision 1.0 / recall
#        1.0 on the 2-success / 8-failure corpus.
# GRASP: eye-view hold episode-mean >= 0.50 gives precision 0.75 (the corpus is
#        harder for grasp; kept as the best measured point, documented honestly).
SUCCESS_CRITERIA = {
    "PLACE_OBJECT": SuccessCriterion(
        skill="PLACE_OBJECT",
        subs=[
            SubQuestion(SUCCESS_QUESTIONS["PLACE_OBJECT_clear"], view="right", threshold=0.75),
            SubQuestion(SUCCESS_QUESTIONS["PLACE_OBJECT_seated"], view="right", threshold=0.50),
        ],
        aggregate="mean",
        min_frames=5,
        note="clear@right>=0.75 (AUC 1.0) AND seated@right>=0.50 guard; prec 1.0 rec 1.0",
    ),
    "GRASP_OBJECT": SuccessCriterion(
        skill="GRASP_OBJECT",
        subs=[SubQuestion(SUCCESS_QUESTIONS["GRASP_OBJECT"], view="eye", threshold=0.50)],
        aggregate="mean",
        min_frames=3,
        note="eye-view hold mean >=0.50 (best measured, precision 0.75)",
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
