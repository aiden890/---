"""Unit tests for the obs-only skill-termination verifier (task t_d4268a2e).

Runs on any CPU box -- no torch, no numpy, no simulator, no model. Uses the
MockVLMBackend and tiny hand-built obs payloads. Covers:

  1. obs-only schema guard: privileged sim predicates are rejected everywhere
     (ObsInput, from_mapping, ObsVLMVerifier.update).
  2. camera-key guard: unknown camera keys rejected; aliases accepted.
  3. proprio gate: settle / gripper-stable / candidate_stop / closed detection.
  4. VLM hysteresis latch: K consecutive yes -> ADVANCE; a 'no' resets the run.
  5. VLM cadence: min-interval gating + event (candidate_stop) gating.
  6. timeout -> REPLAN when the VLM never latches within the step budget.
  7. mock end-to-end: a GRASP-like obs stream terminates at the recognised step.

Run: python3 test_obs_verifier.py   (exit 0 = all pass)
"""
from __future__ import annotations

import sys

from schemas import Decision
from obs_verifier import (CAMERA_KEYS, STATE_DIM, ObsInput, ProprioGate,
                          ObsVLMVerifier, assert_obs_only,
                          FORBIDDEN_PREDICATE_KEYS)
from vlm_backends import MockVLMBackend, compose_three_cam  # noqa: F401

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def _imgs():
    # tiny fake images; obs verifier never inspects pixels (mock backend).
    return {k: [[0]] for k in CAMERA_KEYS}


def _proprio(ee=(0.0, 0.0, 0.0), grip=(0.03, -0.03)):
    v = [0.0] * STATE_DIM
    v[0], v[1], v[2] = ee
    v[6], v[7] = grip
    return v


def main():
    # 1. obs-only schema guard -------------------------------------------- #
    for badkey in ("lid_on_blender", "official_check_success", "lid_pos", "lid_grasped"):
        raised = False
        try:
            assert_obs_only({**_imgs(), badkey: True})
        except ValueError:
            raised = True
        check(f"guard.assert_obs_only_rejects[{badkey}]", raised)

    raised = False
    try:
        ObsInput(images={**_imgs(), "lid_on_blender": True}, proprio=_proprio())
    except ValueError:
        raised = True
    check("guard.ObsInput_rejects_predicate", raised)

    raised = False
    try:
        ObsInput.from_mapping({**_imgs(), "official_check_success": False, "proprio": _proprio()})
    except ValueError:
        raised = True
    check("guard.from_mapping_rejects_predicate", raised)

    # every forbidden key is actually caught
    all_caught = True
    for k in FORBIDDEN_PREDICATE_KEYS:
        try:
            assert_obs_only({k: 0})
            all_caught = False
        except ValueError:
            pass
    check("guard.all_forbidden_keys_caught", all_caught, f"{len(FORBIDDEN_PREDICATE_KEYS)} keys")

    # verifier.update rejects a leaked predicate mid-run
    v = ObsVLMVerifier("GRASP_OBJECT", MockVLMBackend([0.0]), max_steps=50)
    leaked = ObsInput(images=_imgs(), proprio=_proprio())
    object.__setattr__(leaked, "images", {**_imgs(), "lid_grasped": True})  # smuggle
    raised = False
    try:
        v.update(leaked)
    except ValueError:
        raised = True
    check("guard.update_rejects_leaked_predicate", raised)

    # 2. camera-key guard -------------------------------------------------- #
    raised = False
    try:
        ObsInput(images={"not_a_cam": [[0]]}, proprio=None)
    except ValueError:
        raised = True
    check("guard.unknown_camera_rejected", raised)
    ok = ObsInput(images={"video." + CAMERA_KEYS[0]: [[0]],
                          "video." + CAMERA_KEYS[1]: [[0]],
                          "video." + CAMERA_KEYS[2]: [[0]]}, proprio=None)
    check("guard.video_prefixed_camera_accepted", set(ok.images) and True)

    # 3. proprio gate ------------------------------------------------------ #
    g = ProprioGate(settle_eps=1e-2, settle_window=3, grip_eps=1e-2)
    for _ in range(5):
        g.update(_proprio(ee=(0.0, 0.0, 0.0), grip=(0.02, -0.02)))  # stationary
    check("proprio.ee_settled", g.ee_settled())
    check("proprio.gripper_stable", g.gripper_stable())
    check("proprio.candidate_stop", g.candidate_stop())
    check("proprio.gripper_closed_when_small_qpos", g.gripper_closed())

    g2 = ProprioGate(settle_eps=1e-3, settle_window=3)
    for i in range(5):
        g2.update(_proprio(ee=(0.1 * i, 0.0, 0.0)))  # moving fast
    check("proprio.not_settled_when_moving", not g2.ee_settled())

    # 4. hysteresis latch -------------------------------------------------- #
    # schedule: no, no, yes, yes -> latch on the 2nd consecutive yes (k=2).
    be = MockVLMBackend([0.1, 0.2, 0.9, 0.95])
    v = ObsVLMVerifier("GRASP_OBJECT", be, max_steps=100, vlm_min_interval=1,
                       hysteresis_k=2, tau=0.6, event_gated=False)
    decisions = []
    for s in range(6):
        r = v.update(ObsInput(images=_imgs(), proprio=_proprio(), step=s))
        decisions.append(r.decision)
    check("hysteresis.advances_on_2nd_consec_yes", Decision.ADVANCE in decisions)
    adv_idx = decisions.index(Decision.ADVANCE)
    check("hysteresis.advance_at_step4", adv_idx == 3, f"advance at update #{adv_idx+1}")
    check("hysteresis.records_success_step", v.succeeded_step == 4, str(v.succeeded_step))

    # a 'no' in the middle resets the consecutive-yes run (no latch).
    be = MockVLMBackend([0.9, 0.1, 0.9, 0.1, 0.9])
    v = ObsVLMVerifier("PLACE_OBJECT", be, max_steps=100, vlm_min_interval=1,
                       hysteresis_k=2, tau=0.6, event_gated=False)
    latched = False
    for s in range(5):
        if v.update(ObsInput(images=_imgs(), proprio=_proprio(), step=s)).decision is Decision.ADVANCE:
            latched = True
    check("hysteresis.no_resets_run", not latched, "alternating yes/no must not latch")

    # 5. cadence ----------------------------------------------------------- #
    # min-interval 16, event gating off: exactly ceil(steps/16) calls.
    be = MockVLMBackend([0.0])
    v = ObsVLMVerifier("MOVE_OBJECT", be, max_steps=100, vlm_min_interval=16,
                       hysteresis_k=1, tau=0.6, event_gated=False)
    for s in range(48):
        v.update(ObsInput(images=_imgs(), proprio=_proprio(), step=s))
    check("cadence.min_interval_limits_calls", be.calls == 3, f"{be.calls} calls in 48 steps @16")

    # event gating: a candidate_stop triggers a query before the interval.
    be = MockVLMBackend([0.0])
    gate = ProprioGate(settle_eps=1e-2, settle_window=2, grip_eps=1e-2)
    v = ObsVLMVerifier("GRASP_OBJECT", be, max_steps=100, vlm_min_interval=1000,
                       hysteresis_k=1, tau=0.6, event_gated=True, proprio_gate=gate)
    # first query happens at step 1 (interval elapsed since -inf); feed motion
    # so no candidate_stop, then settle to trigger an event query.
    v.update(ObsInput(images=_imgs(), proprio=_proprio(ee=(0.0, 0.0, 0.0)), step=0))
    calls_after_first = be.calls
    for s in range(1, 6):
        v.update(ObsInput(images=_imgs(), proprio=_proprio(ee=(0.0, 0.0, 0.0)), step=s))
    check("cadence.event_gate_triggers_extra_query", be.calls > calls_after_first,
          f"{be.calls} calls (event-gated)")

    # 6. timeout -> REPLAN ------------------------------------------------- #
    be = MockVLMBackend([0.0])  # VLM never says yes
    v = ObsVLMVerifier("GRASP_OBJECT", be, max_steps=10, vlm_min_interval=1,
                       hysteresis_k=1, tau=0.6, event_gated=False)
    last = None
    for s in range(10):
        last = v.update(ObsInput(images=_imgs(), proprio=_proprio(), step=s))
    check("timeout.replan_when_never_latched", last.decision is Decision.REPLAN, last.reason)

    # 7. mock end-to-end GRASP stream ------------------------------------- #
    # arm moves for 20 steps, then settles + gripper closes; VLM recognises
    # completion (yes) only once settled. Verifier should ADVANCE shortly after.
    def schedule(call_idx, q):
        # backend can't see step; use call index. Early calls are pre-settle.
        return 0.05 if call_idx < 2 else 0.92
    be = MockVLMBackend(schedule)
    gate = ProprioGate(settle_eps=8e-3, settle_window=3, grip_eps=3e-3)
    v = ObsVLMVerifier("GRASP_OBJECT", be, max_steps=208, vlm_min_interval=16,
                       hysteresis_k=2, tau=0.6, event_gated=True, proprio_gate=gate)
    adv_step = None
    for s in range(60):
        moving = s < 20
        ee = (0.02 * s, 0.0, 0.0) if moving else (0.4, 0.0, 0.0)
        grip = (0.05, -0.05) if moving else (0.02, -0.02)
        r = v.update(ObsInput(images=_imgs(), proprio=_proprio(ee=ee, grip=grip), step=s))
        if r.decision is Decision.ADVANCE:
            adv_step = s + 1
            break
    check("e2e.grasp_stream_advances", adv_step is not None, f"advanced at step {adv_step}")
    check("e2e.advance_after_settle", adv_step is not None and adv_step > 20,
          f"advanced at {adv_step} (settle at 20)")
    check("e2e.reasonable_vlm_budget", be.calls <= 8, f"{be.calls} VLM calls over {adv_step} steps")

    # ---- verdict --------------------------------------------------------- #
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results)-n_fail}/{len(results)} checks passed.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
