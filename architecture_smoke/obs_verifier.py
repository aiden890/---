"""Observation-only skill-termination verifier (task t_d4268a2e).

The parent ``verifier.PredicateVerifier`` judges skill termination from the
simulator's ground-truth predicates (``skill_eval.Sim.predicates()`` ->
``lid_on_blender``, ``official_check_success``, ``lid_pos`` ...). Those are
*privileged* object-pose / fixture states the real robot can NEVER observe, so a
predicate verifier cannot transfer to hardware.

This module re-designs the verifier so skill termination is judged **only from
what the robot actually receives**:

  * 3 camera images -- ``robot0_agentview_left``, ``robot0_agentview_right``,
    ``robot0_eye_in_hand`` (the same ``rollout.CAMERA_KEYS``), and
  * the 14-D proprioceptive state (``rollout.observation_to_state``): EE
    position (relative) + EE rotation (axis-angle) + gripper_qpos + base
    position + base rotation.

Design (operator decision: **the VLM is the judge**; proprio is auxiliary):

  1. ``ProprioGate`` -- a cheap, obs-only event detector over the 14-D state.
     It flags when the arm has *settled* (EE nearly stationary) and the gripper
     state is *stable*. This is NOT the success judge; it decides *when it is
     worth asking the VLM*, so the expensive VLM is not run every 20 Hz step
     (unrealistic latency/cost). The proprio-only success heuristic is also
     exposed as a baseline/comparison channel.

  2. ``ObsVLMVerifier`` -- the real judge. On a realistic cadence (a minimum
     step interval AND/OR a proprio ``candidate_stop`` event) it poses a single
     yes/no visual question about *this skill's* completion to a vision-language
     model and reads P(yes). A hysteresis latch (K consecutive confident "yes")
     turns the skill SUCCESS -> ADVANCE, so the skill stops the moment its goal
     is recognised (no over-run). Budget exhaustion -> REPLAN.

HARD INVARIANT: the runtime judge input is obs only. ``ObsInput.from_mapping``
and ``assert_obs_only`` reject any privileged simulator predicate key
(``lid_on_blender``, ``official_check_success``, ``lid_pos``, ...). Privileged
predicates may be used ONLY as *offline* supervision labels in the validation
harness, never inside this verifier -- enforced here in code.

The VQA prompt text and the logit->probability math are reused verbatim from the
policy's own ``vlm_scorer`` module (single source of truth for the VQA), so the
real backend adds no second copy of the prompt/logit code.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from schemas import Decision, VerificationResult

# --------------------------------------------------------------------------- #
#  Obs contract -- the ONLY thing the runtime judge may read                    #
# --------------------------------------------------------------------------- #
# The three camera streams the policy itself consumes (rollout.CAMERA_KEYS,
# with or without the "video." prefix used by the LeRobot obs dict).
CAMERA_KEYS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)
_CAMERA_ALIASES = CAMERA_KEYS + tuple("video." + k for k in CAMERA_KEYS)

# 14-D proprio layout (rollout.observation_to_state, EE-first):
#   [0:3]  end_effector_position_relative
#   [3:6]  end_effector_rotation (axis-angle)
#   [6:8]  gripper_qpos (two finger joints)
#   [8:11] base_position
#   [11:14] base_rotation (axis-angle)
EE_POS = slice(0, 3)
EE_ROT = slice(3, 6)
GRIPPER_QPOS = slice(6, 8)
BASE_POS = slice(8, 11)
BASE_ROT = slice(11, 14)
STATE_DIM = 14

# Privileged simulator predicates that the real robot cannot observe. Any of
# these appearing in a runtime obs payload is a leak and must fail loudly.
FORBIDDEN_PREDICATE_KEYS = frozenset({
    "lid_grasped", "lid_lifted", "lid_on_blender", "lid_on_counter",
    "in_preplace_region", "official_check_success", "lid_upright_7deg",
    "gripper_lid_far_0.15", "gripper_lid_contact", "lid_pos",
    "lid_dz_from_rest", "lid_xy_to_closed_pos", "lid_dz_to_closed_pos",
    "lid_dist_to_closed_pos", "lid_other_contacts", "eef_lid_dist",
    "rest_lid_pos", "lid_closed_pos",
})


def assert_obs_only(payload: Mapping[str, Any]) -> None:
    """Raise if any privileged simulator predicate key is present.

    Guards the runtime judge input schema: only camera images + proprio are
    allowed to reach the verifier. Privileged sim predicates are offline
    supervision labels, never verifier inputs.
    """
    leaked = sorted(k for k in payload if k in FORBIDDEN_PREDICATE_KEYS)
    if leaked:
        raise ValueError(
            "obs-only verifier received privileged simulator predicate(s) "
            f"{leaked}; runtime input must be camera images + proprio only "
            "(privileged predicates are offline supervision labels)"
        )


@dataclass(frozen=True)
class ObsInput:
    """Exactly what the robot receives at one step: images + 14-D proprio.

    ``images`` maps camera key -> HxWx3 uint8 array (or any array-like). The
    frozen dataclass has NO field for predicates, so a privileged predicate can
    never be smuggled through the typed path. ``from_mapping`` additionally
    scans loose dicts and rejects privileged keys.
    """

    images: Mapping[str, Any]
    proprio: Optional[Sequence[float]]
    step: int = 0

    def __post_init__(self):
        assert_obs_only(self.images)
        bad_cam = [k for k in self.images if k not in _CAMERA_ALIASES]
        if bad_cam:
            raise ValueError(f"unknown camera key(s) {sorted(bad_cam)}; "
                             f"allowed: {list(CAMERA_KEYS)}")
        if self.proprio is not None and len(self.proprio) != STATE_DIM:
            raise ValueError(f"proprio must be {STATE_DIM}-D, got {len(self.proprio)}")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], step: int = 0) -> "ObsInput":
        """Build an ObsInput from a loose obs dict, rejecting privileged keys."""
        assert_obs_only(payload)
        images = {k: v for k, v in payload.items() if k in _CAMERA_ALIASES}
        proprio = payload.get("proprio")
        return cls(images=images, proprio=proprio, step=step)

    def camera(self, key: str):
        for cand in (key, "video." + key):
            if cand in self.images:
                return self.images[cand]
        raise KeyError(f"camera {key!r} not in obs (have {sorted(self.images)})")


# --------------------------------------------------------------------------- #
#  Proprio gate -- obs-only event detector (auxiliary + VLM cadence gate)       #
# --------------------------------------------------------------------------- #
def _l2(a, b) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) ** 0.5


@dataclass
class ProprioGate:
    """Detects arm-settled / gripper-stable events from the 14-D proprio stream.

    Pure obs. Used to (a) gate the VLM to realistic moments (arm stopped) and
    (b) provide a proprio-only success heuristic as a comparison baseline.

    Thresholds are in the proprio's own units (relative EE metres, finger qpos).
    """

    settle_eps: float = 6e-3          # EE moved < this over the settle window -> settled
    settle_window: int = 4            # consecutive settled steps to call it settled
    grip_eps: float = 2e-3            # finger qpos change < this -> gripper stable
    grip_closed_thresh: float = 0.05  # |finger_qpos| sum below -> fingers closed on object
    grip_open_thresh: float = 0.06    # finger separation above -> open

    def __post_init__(self):
        self._prev_ee = None
        self._prev_grip = None
        self._settled_run = 0
        self._grip_stable_run = 0
        self.ee_speed = None
        self.grip_delta = None

    def update(self, proprio: Optional[Sequence[float]]) -> None:
        if proprio is None:
            return
        ee = list(proprio)[EE_POS]
        grip = list(proprio)[GRIPPER_QPOS]
        if self._prev_ee is not None:
            self.ee_speed = _l2(ee, self._prev_ee)
            self._settled_run = self._settled_run + 1 if self.ee_speed < self.settle_eps else 0
        if self._prev_grip is not None:
            self.grip_delta = _l2(grip, self._prev_grip)
            self._grip_stable_run = self._grip_stable_run + 1 if self.grip_delta < self.grip_eps else 0
        self._prev_ee = ee
        self._prev_grip = grip
        self._last_grip = grip

    # ---- obs-only signals ------------------------------------------------- #
    def ee_settled(self) -> bool:
        return self._settled_run >= self.settle_window

    def gripper_stable(self) -> bool:
        return self._grip_stable_run >= self.settle_window

    def gripper_closed(self) -> bool:
        g = getattr(self, "_last_grip", None)
        if g is None:
            return False
        # finger qpos are ~ +open/-open symmetric; small magnitude sum -> closed.
        return (abs(float(g[0])) + abs(float(g[1]))) < self.grip_closed_thresh

    def candidate_stop(self) -> bool:
        """A realistic moment to ask the VLM: arm settled and gripper stable."""
        return self.ee_settled() and self.gripper_stable()


# --------------------------------------------------------------------------- #
#  VLM backend interface                                                        #
# --------------------------------------------------------------------------- #
class VLMBackend:
    """score(images, question_text) -> P(yes) in [0, 1].

    ``score_view(images, question_text, view)`` optionally routes the VQA to a
    SINGLE camera view ('left'|'right'|'eye') instead of the wide 3-cam concat
    ('full'). Calibration (task t_32a4f9b6) found a single view separates the
    strict success judge far better than the diluted concat. Backends that do
    not override ``score_view`` fall back to the full-concat ``score``.
    """

    def score(self, images: Mapping[str, Any], question_text: str) -> float:
        raise NotImplementedError

    def score_view(self, images: Mapping[str, Any], question_text: str,
                   view: str = "full") -> float:
        # default: ignore the view request, score the full concat.
        return self.score(images, question_text)


# Per-skill completion questions. Each asks a single yes/no visual fact the real
# robot cameras can answer -- NO privileged pose is referenced. Kept here as the
# task binding (obs_verifier is CloseBlenderLid-bound like bindings.py).
SKILL_QUESTIONS = {
    "GRASP_OBJECT": (
        "These are camera views of a robot manipulation scene. Is the robot "
        "gripper firmly grasping and holding the blender lid, having lifted it "
        "clear off the counter? Answer yes or no."
    ),
    "MOVE_OBJECT": (
        "These are camera views of a robot manipulation scene. Is the robot "
        "holding the blender lid directly above the blender base, positioned "
        "and ready to place it down? Answer yes or no."
    ),
    "PLACE_OBJECT": (
        "These are camera views of a robot manipulation scene showing a "
        "blender. Has the blender lid been placed back on top of the blender "
        "base so the blender is closed, AND has the robot gripper let go of the "
        "lid and moved away from it? Answer yes or no."
    ),
}


# --------------------------------------------------------------------------- #
#  The obs-only VLM verifier                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class ObsVerdict:
    decision: Decision
    reason: str
    step: int
    n_vlm_calls: int
    last_prob: Optional[float]
    consecutive_yes: int
    vlm_queried: bool
    proprio_candidate: bool
    proprio_success_heuristic: bool


class ObsVLMVerifier:
    """Judge one skill's termination from obs only, VLM as the arbiter.

    Cadence: the VLM is queried when ``elapsed`` reaches a multiple of
    ``vlm_min_interval`` (a realistic re-plan-boundary cadence) OR, when
    ``event_gated``, whenever the proprio gate reports a ``candidate_stop``
    (arm settled + gripper stable) -- so a skill that finishes early is caught
    without paying for a VLM call every step.

    Latch: ``hysteresis_k`` consecutive queries with P(yes) >= ``tau`` -> the
    skill is done -> ADVANCE. The consecutive counter resets on any confident
    "no", so a transient false positive cannot latch.

    Budget: ``max_steps`` without a latch -> REPLAN (timeout).
    """

    def __init__(self, skill_name: str, backend: VLMBackend, *,
                 max_steps: int, question_text: Optional[str] = None,
                 vlm_min_interval: int = 16, hysteresis_k: int = 2,
                 tau: float = 0.6, event_gated: bool = True,
                 proprio_gate: Optional[ProprioGate] = None,
                 view: str = "full", sequence_model=None):
        self.skill_name = skill_name
        self.backend = backend
        self.max_steps = max_steps
        self.question_text = question_text or SKILL_QUESTIONS[skill_name]
        self.vlm_min_interval = max(1, int(vlm_min_interval))
        self.hysteresis_k = max(1, int(hysteresis_k))
        self.tau = float(tau)
        self.event_gated = bool(event_gated)
        if view not in ("full", "left", "right", "eye"):
            raise ValueError(f"invalid verifier view: {view}")
        self.view = view
        if sequence_model is not None:
            from sequence_boundary import SequenceBoundaryModel
            self.sequence_model = (sequence_model if isinstance(sequence_model, SequenceBoundaryModel)
                                   else SequenceBoundaryModel.from_json(sequence_model))
            if self.sequence_model.view != self.view:
                raise ValueError(
                    f"sequence model expects view={self.sequence_model.view!r}, "
                    f"verifier configured with view={self.view!r}")
        else:
            self.sequence_model = None
        self.gate = proprio_gate if proprio_gate is not None else ProprioGate()

        self.elapsed = 0
        self.consecutive_yes = 0
        self.n_vlm_calls = 0
        self.last_prob: Optional[float] = None
        self.succeeded_step: Optional[int] = None
        self._last_query_step = -10 ** 9
        self.vlm_seconds = 0.0
        self.query_log: list[dict] = []
        self.sequence_probability: Optional[float] = None

    def _should_query(self) -> bool:
        due = (self.elapsed - self._last_query_step) >= self.vlm_min_interval
        event = self.event_gated and self.gate.candidate_stop() and \
            (self.elapsed - self._last_query_step) >= 1
        return bool(due or event)

    def update(self, obs: ObsInput) -> VerificationResult:
        """One executed step. Obs-only; raises on any privileged-key leak."""
        assert_obs_only(obs.images)
        self.elapsed += 1
        self.gate.update(obs.proprio)

        queried = False
        if self._should_query():
            queried = True
            t0 = time.time()
            prob = float(self.backend.score_view(obs.images, self.question_text,
                                                 view=self.view))
            self.vlm_seconds += time.time() - t0
            self.n_vlm_calls += 1
            self._last_query_step = self.elapsed
            self.last_prob = prob
            yes = prob >= self.tau
            self.consecutive_yes = self.consecutive_yes + 1 if yes else 0
            sequence_latched = False
            if self.sequence_model is not None:
                sequence_scores = {self.view: prob}
                for sequence_view in self.sequence_model.views:
                    if sequence_view == self.view:
                        continue
                    extra_t0 = time.time()
                    sequence_scores[sequence_view] = float(self.backend.score_view(
                        obs.images, self.question_text, view=sequence_view))
                    self.vlm_seconds += time.time() - extra_t0
                    self.n_vlm_calls += 1
                sequence_score = self.sequence_model.aggregate_scores(sequence_scores)
                sequence_latched, self.sequence_probability = self.sequence_model.update(
                    sequence_score, obs.proprio, self.elapsed)
            self.query_log.append({"step": self.elapsed, "prob": round(prob, 4),
                                   "yes": yes, "consec": self.consecutive_yes,
                                   "sequence_probability": self.sequence_probability,
                                   "sequence_latched": sequence_latched})

        proprio_success = self.gate.candidate_stop() and self.gate.gripper_closed()
        latched = (self.sequence_model.positive_run >= self.sequence_model.dwell
                   if self.sequence_model is not None
                   else self.consecutive_yes >= self.hysteresis_k)

        if latched:
            if self.succeeded_step is None:
                self.succeeded_step = self.elapsed
            if self.sequence_model is not None:
                reason = (
                    f"sequence verifier recognised '{self.skill_name}' complete: "
                    f"P(boundary)={self.sequence_probability:.3f} >= "
                    f"tau={self.sequence_model.tau} for "
                    f"{self.sequence_model.dwell} consecutive query(ies)")
            else:
                reason = (
                    f"VLM recognised '{self.skill_name}' complete: P(yes)={self.last_prob:.3f} "
                    f">= tau={self.tau} for {self.hysteresis_k} consecutive query(ies)")
            return self._result(Decision.ADVANCE, reason, queried, proprio_success)

        if self.elapsed >= self.max_steps:
            return self._result(
                Decision.REPLAN,
                f"step budget {self.max_steps} exhausted without VLM latch "
                f"(last P(yes)={self.last_prob})",
                queried, proprio_success)

        return self._result(Decision.CONTINUE, "skill not yet recognised as complete",
                            queried, proprio_success)

    def _result(self, decision: Decision, reason: str, queried: bool,
                proprio_success: bool) -> VerificationResult:
        # 'predicates' field carries obs-derived diagnostics ONLY (no sim state).
        diag = {
            "judge": "obs_vlm",
            "vlm_prob": self.last_prob,
            "consecutive_yes": self.consecutive_yes,
            "n_vlm_calls": self.n_vlm_calls,
            "vlm_queried_this_step": queried,
            "proprio_ee_settled": self.gate.ee_settled(),
            "proprio_gripper_closed": self.gate.gripper_closed(),
            "proprio_candidate_stop": self.gate.candidate_stop(),
            "proprio_success_heuristic": proprio_success,
            "sequence_probability": self.sequence_probability,
            "sequence_positive_run": (self.sequence_model.positive_run
                                      if self.sequence_model is not None else None),
        }
        hold = (self.sequence_model.positive_run if self.sequence_model is not None
                else self.consecutive_yes)
        return VerificationResult(decision=decision, reason=reason,
                                  hold=hold, elapsed=self.elapsed,
                                  predicates=diag)

    # convenience for harness logging
    def stats(self) -> dict:
        return {
            "skill": self.skill_name, "view": self.view,
            "tau": self.tau, "hysteresis_k": self.hysteresis_k,
            "vlm_min_interval": self.vlm_min_interval,
            "elapsed": self.elapsed,
            "n_vlm_calls": self.n_vlm_calls, "vlm_seconds": round(self.vlm_seconds, 4),
            "sequence_model": self.sequence_model is not None,
            "sequence_probability": self.sequence_probability,
            "succeeded_step": self.succeeded_step,
            "mean_vlm_latency_ms": round(1000 * self.vlm_seconds / self.n_vlm_calls, 2)
            if self.n_vlm_calls else None,
        }
