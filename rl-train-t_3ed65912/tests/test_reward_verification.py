"""Deep reward/predicate-alignment verification for CloseBlenderLid GRPO (task t_3ed65912).

This is the operator-mandated GATE that must fully pass BEFORE any training. It goes
deeper than the env card's test_env_components.py: it checks reward/predicate ALIGNMENT
on positive AND negative cases, milestone gating + once-only under oscillation, penalty
triggering ONLY in the right situation, and the specific t_4a072806 PLACE edge case
(lid seated then tilts on release -> official success must flip to False and terminal
reward must NOT be paid at the tilted step). Reward code = the env card's RewardManager
(single source of truth); this only VERIFIES it.

Predicate keys mirror rollouts-xiaomi-t_4a072806/tools/skill_eval.py Sim.predicates():
  lid_grasped, lid_lifted, in_preplace_region, lid_on_blender, gripper_lid_contact,
  gripper_lid_far_0.15, lid_upright_7deg, official_check_success, lid_other_contacts.

Run (client image, numpy only): python3 tests/test_reward_verification.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/rl_env/src")
# local fallback for running outside the container
_LOCAL = Path(__file__).resolve().parent.parent.parent / "rl-env-t_4f3f2b20" / "src"
if _LOCAL.exists():
    sys.path.insert(0, str(_LOCAL))

from reward import (  # noqa: E402
    RewardConfig, RewardManager, official_success, SKILL_MILESTONES, DEFAULT_MILESTONE_BONUS,
)

_FAILS = []


def check(cond, msg):
    if cond:
        print(f"[ok] {msg}")
    else:
        print(f"[FAIL] {msg}")
        _FAILS.append(msg)


def _p(**kw):
    base = {
        "lid_grasped": False, "lid_lifted": False, "in_preplace_region": False,
        "lid_on_blender": False, "gripper_lid_contact": False, "gripper_lid_far_0.15": False,
        "lid_upright_7deg": True, "official_check_success": False, "lid_other_contacts": [],
    }
    if "gripper_far" in kw:
        base["gripper_lid_far_0.15"] = kw.pop("gripper_far")
    base.update(kw)
    return base


# ------------------------------------------------------------------ #
# 1. Terminal reward EXACTLY tracks the official predicate (pos + neg).
# ------------------------------------------------------------------ #
def test_terminal_positive_matches_official():
    rm = RewardManager(RewardConfig())
    # official success true -> terminal paid (>0), success flag true
    rb = rm.step_reward(50, _p(lid_on_blender=True, gripper_far=True, official_check_success=True))
    check(rb.terminal > 0 and rb.success, "terminal reward fires exactly when official_check_success=True")


def test_terminal_negative_no_reward_on_partial():
    # every partial-but-not-official state must yield ZERO terminal reward
    partials = [
        _p(lid_on_blender=True, gripper_far=False, official_check_success=False),   # seated but gripper near
        _p(lid_on_blender=False, gripper_far=True, official_check_success=False),   # gripper far but not seated
        _p(lid_grasped=True, lid_lifted=True, official_check_success=False),        # mid-grasp
        _p(in_preplace_region=True, official_check_success=False),                  # pre-place only
    ]
    ok = True
    for p in partials:
        rm = RewardManager(RewardConfig())
        rb = rm.step_reward(30, p)
        if rb.terminal != 0.0 or rb.success:
            ok = False
    check(ok, "terminal reward is ZERO for every partial/non-official predicate state")


def test_terminal_paid_once_even_if_success_persists():
    rm = RewardManager(RewardConfig())
    paid = 0
    for s in range(1, 20):
        rb = rm.step_reward(s, _p(lid_on_blender=True, gripper_far=True, official_check_success=(s >= 5)))
        if rb.terminal > 0:
            paid += 1
    check(paid == 1, "terminal reward paid exactly once even though success persists many steps")


def test_terminal_matches_manual_predicate_formula():
    # official_success() fallback must equal lid_on_blender AND gripper_far AND upright
    # (operator spec: lid_on_blender=True AND gripper-lid dist>0.15m). Drop the env key.
    def q(**kw):
        p = _p(**kw); p.pop("official_check_success"); return p
    check(official_success(q(lid_on_blender=True, gripper_far=True, lid_upright_7deg=True)) is True,
          "official_success TRUE iff lid_on_blender AND gripper_far AND upright")
    fails = [
        official_success(q(lid_on_blender=False, gripper_far=True, lid_upright_7deg=True)),
        official_success(q(lid_on_blender=True, gripper_far=False, lid_upright_7deg=True)),
        official_success(q(lid_on_blender=True, gripper_far=True, lid_upright_7deg=False)),
    ]
    check(not any(fails), "official_success FALSE if ANY conjunct (seated/far/upright) is false")


# ------------------------------------------------------------------ #
# 2. PLACE tilt-on-release edge case (the t_4a072806 PLACE scenario).
# ------------------------------------------------------------------ #
def test_place_tilt_on_release_no_false_reward():
    """Lid gets seated (lid_on_blender True) while gripper still near, then on release
    the lid tilts (upright False) so official success is FALSE. Terminal reward must
    NEVER be paid across this trajectory, and success must read False at the tilted step."""
    rm = RewardManager(RewardConfig())
    traj = [
        _p(lid_grasped=True, lid_lifted=True),                                   # holding
        _p(lid_grasped=True, in_preplace_region=True),                           # pre-place
        _p(lid_on_blender=True, gripper_lid_contact=True, official_check_success=False),  # seated, gripper near
        _p(lid_on_blender=True, gripper_far=True, lid_upright_7deg=False, official_check_success=False),  # release+tilt
    ]
    total_terminal = 0.0
    last = None
    for s, p in enumerate(traj, 1):
        rb = rm.step_reward(s, p, done=(s == len(traj)))
        total_terminal += rb.terminal
        last = rb
    check(total_terminal == 0.0, "PLACE tilt-on-release: terminal reward NEVER paid (official stayed False)")
    check(last.success is False, "PLACE tilt-on-release: success flag False at the tilted release step")


def test_place_clean_release_pays_at_retreat_step():
    """Correct timing: terminal reward is paid on the exact step official flips True."""
    rm = RewardManager(RewardConfig())
    traj = [
        (_p(lid_on_blender=True, gripper_lid_contact=True, official_check_success=False), False),  # seated, near
        (_p(lid_on_blender=True, gripper_far=True, lid_upright_7deg=True, official_check_success=True), True),  # retreat -> success
        (_p(lid_on_blender=True, gripper_far=True, lid_upright_7deg=True, official_check_success=True), True),  # still success
    ]
    paid_step = None
    for s, (p, _) in enumerate(traj, 1):
        rb = rm.step_reward(s, p)
        if rb.terminal > 0 and paid_step is None:
            paid_step = s
    check(paid_step == 2, "terminal reward is paid on the exact step official success first becomes True")


# ------------------------------------------------------------------ #
# 3. Milestones: exact gating, once-only, no early/duplicate firing.
# ------------------------------------------------------------------ #
def test_each_milestone_fires_only_on_its_predicate():
    conds = {
        "stable_grasp": _p(lid_grasped=True),
        "lid_lifted": _p(lid_lifted=True),
        "collision_free_transfer": _p(lid_grasped=True),                      # grasped & no other contact
        "pre_place_reached": _p(in_preplace_region=True),
        "lid_seated": _p(lid_on_blender=True),
        "released": _p(lid_on_blender=True, gripper_lid_contact=False),
        "retreated": _p(lid_on_blender=True, gripper_far=True),
    }
    ok = True
    for name, fn in SKILL_MILESTONES.items():
        if name == "grasp_maintained":
            continue
        want = conds[name]
        if not fn(want):
            ok = False; print(f"   milestone {name} did not fire on its own condition")
        # negative: empty predicate must NOT fire it
        if fn(_p()):
            ok = False; print(f"   milestone {name} fired on the empty/false state (early firing)")
    check(ok, "each milestone fires ONLY under its exact predicate, never on the empty state")


def test_milestone_once_only_under_oscillation():
    """Grasp toggles true/false/true; stable_grasp must be paid exactly once (no farming)."""
    rm = RewardManager(RewardConfig())
    seq = [False, True, False, True, True, False, True]
    fires = 0
    for s, g in enumerate(seq, 1):
        rb = rm.step_reward(s, _p(lid_grasped=g))
        fires += rb.milestones_fired.count("stable_grasp")
    check(fires == 1, "milestone paid exactly once despite predicate oscillating true/false/true")


def test_no_milestone_before_condition():
    rm = RewardManager(RewardConfig())
    early_fires = 0
    for s in range(1, 6):
        rb = rm.step_reward(s, _p())  # nothing true yet
        early_fires += len(rb.milestones_fired)
    check(early_fires == 0, "no milestone is paid before its predicate is ever true")


def test_milestones_disabled_variant():
    rm = RewardManager(RewardConfig(use_milestones=False))
    rb = rm.step_reward(5, _p(lid_grasped=True, lid_lifted=True))
    check(rb.milestone == 0.0 and not rb.milestones_fired,
          "reward-variant terminal-only: milestones contribute nothing")


# ------------------------------------------------------------------ #
# 4. Penalties trigger ONLY in their exact situation.
# ------------------------------------------------------------------ #
def test_object_drop_penalty_only_when_dropped():
    # was grasped at t1, then not grasped/seated/contact at t2 -> drop penalty
    rm = RewardManager(RewardConfig())
    rm.step_reward(1, _p(lid_grasped=True))
    rb = rm.step_reward(2, _p(lid_grasped=False))
    check(rb.penalty <= DEFAULT_MILESTONE_BONUS.get("x", 0) + (-0.25) + 1e-9 and rb.penalty < 0,
          "object_dropped penalty fires when a held lid is lost mid-air")

    # placed (seated) after grasp must NOT be treated as a drop
    rm2 = RewardManager(RewardConfig())
    rm2.step_reward(1, _p(lid_grasped=True))
    rb2 = rm2.step_reward(2, _p(lid_grasped=False, lid_on_blender=True))
    check(rb2.penalty == 0.0, "NO drop penalty when the lid was released onto the blender (seated)")

    # still in contact (not fully dropped) must NOT penalize
    rm3 = RewardManager(RewardConfig())
    rm3.step_reward(1, _p(lid_grasped=True))
    rb3 = rm3.step_reward(2, _p(lid_grasped=False, gripper_lid_contact=True))
    check(rb3.penalty == 0.0, "NO drop penalty while the lid is still in gripper contact")


def test_collision_penalty_only_on_contacts():
    # default: collision penalty requires the lid to be GRASPED (carrying), so a resting
    # lid touching its support surface is NOT penalized (avoids dense per-step saturation).
    rm = RewardManager(RewardConfig())
    rb_rest = rm.step_reward(1, _p(lid_other_contacts=["counter"], lid_grasped=False))
    rb_carry = rm.step_reward(2, _p(lid_other_contacts=["counter"], lid_grasped=True))
    check(rb_rest.penalty == 0.0, "NO collision penalty for a resting (ungrasped) lid touching support")
    check(rb_carry.penalty < 0.0, "collision penalty fires when a GRASPED lid hits something")
    # legacy mode (collision_requires_grasp=False): any contact penalized
    rm2 = RewardManager(RewardConfig(collision_requires_grasp=False))
    rb0 = rm2.step_reward(1, _p(lid_other_contacts=[]))
    rb1 = rm2.step_reward(2, _p(lid_other_contacts=["counter"]))
    check(rb0.penalty == 0.0 and rb1.penalty < 0.0,
          "legacy collision mode: penalty fires on ANY non-empty lid_other_contacts")


def test_timeout_penalty_only_on_failed_end():
    rm = RewardManager(RewardConfig())
    mid = rm.step_reward(5, _p())                       # not done -> no timeout penalty
    end_fail = rm.step_reward(6, _p(), truncated=True)  # failed end -> timeout penalty
    check(mid.penalty == 0.0, "no timeout penalty mid-episode")
    check(end_fail.penalty < 0.0, "timeout penalty applied on a truncated/failed episode end")

    rm2 = RewardManager(RewardConfig())
    end_ok = rm2.step_reward(6, _p(lid_on_blender=True, gripper_far=True, official_check_success=True), done=True)
    # success end: terminal>0 and NO timeout penalty
    check(end_ok.penalty == 0.0 and end_ok.terminal > 0,
          "NO timeout penalty on a SUCCESSFUL episode end (only terminal reward)")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"\n{len(fns)} verification groups run; {len(_FAILS)} FAILED.")
    if _FAILS:
        print("FAILURES:")
        for m in _FAILS:
            print("  -", m)
        sys.exit(1)
    print("ALL REWARD-VERIFICATION CHECKS PASSED.")


if __name__ == "__main__":
    _run_all()
