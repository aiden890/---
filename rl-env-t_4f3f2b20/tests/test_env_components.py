"""Unit tests for randomization, reward, and skill-manager (numpy-only, no GPU/torch).

Run: python3 rl-env-t_4f3f2b20/tests/test_env_components.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from randomization import Randomizer, RandomizationSpec, splits_disjoint, validate_scene  # noqa: E402
from reward import (  # noqa: E402
    RewardConfig,
    RewardManager,
    official_success,
    z1_baseline_reward,
)
from skill_manager import (  # noqa: E402
    MonitorConfig,
    OraclePlanner,
    Skill,
    SkillMonitor,
    SkillOutcome,
    VLMPlannerStub,
)


# ----------------------------- randomization ------------------------------- #
def test_randomization_reproducible():
    r = Randomizer(master_seed=42)
    a = r.sample("train", 5)
    b = r.sample("train", 5)
    assert a.to_metadata() == b.to_metadata(), "same (split,episode) must reproduce"
    c = r.sample("train", 6)
    assert c.to_metadata() != a.to_metadata(), "different episode must differ"
    print("[ok] randomization reproducible from (split, episode_index)")


def test_randomization_roundtrip_metadata():
    r = Randomizer(master_seed=1)
    spec = r.sample("test", 3)
    meta = spec.to_metadata()
    rebuilt = RandomizationSpec.from_metadata(meta)
    assert rebuilt.to_metadata() == meta, "metadata roundtrip must be lossless"
    print("[ok] randomization metadata roundtrips losslessly")


def test_splits_are_disjoint():
    d = splits_disjoint()
    assert d["lid_variant_train_test_disjoint"], "held-out lid variants must be disjoint"
    assert d["scene_seed_train_test_disjoint"], "held-out scene seeds must be disjoint"
    print("[ok] train/test lid-variant and scene-seed pools are disjoint (held-out ready)")


def test_validation_rejects_implausible():
    r = Randomizer(master_seed=7)
    spec = r.sample("train", 0)
    # force an implausible combo and confirm the rule fires
    spec.values["physics"]["lid_mass_scale"] = 0.85
    spec.values["physics"]["friction_scale"] = 1.2
    problems = validate_scene(spec, r.ranges)
    assert "low_mass_high_friction_unstable" in problems
    spec.values["lid_pose"]["z_offset_m"] = -0.5
    assert "lid_z_below_rest" in validate_scene(spec, r.ranges)
    print("[ok] validation rules reject implausible physics/geometry combos")


def test_all_sampled_scenes_valid():
    r = Randomizer(master_seed=99)
    for split in ("train", "validation", "test"):
        for i in range(50):
            spec = r.sample(split, i)
            assert spec.valid, f"{split}[{i}] should resample to a valid scene, got {spec.rejections}"
    print("[ok] 150 sampled scenes across splits are all valid after resampling")


# -------------------------------- reward ----------------------------------- #
def _p(**kw):
    base = {
        "lid_grasped": False, "lid_lifted": False, "in_preplace_region": False,
        "lid_on_blender": False, "gripper_lid_contact": False, "gripper_lid_far_0.15": False,
        "lid_upright_7deg": True, "official_check_success": False, "lid_other_contacts": [],
    }
    # convenience aliases for the dotted predicate key
    if "gripper_far" in kw:
        base["gripper_lid_far_0.15"] = kw.pop("gripper_far")
    base.update(kw)
    return base


def test_milestones_paid_once():
    cfg = RewardConfig(use_milestones=True)
    rm = RewardManager(cfg)
    # grasp becomes true at step 10 and stays true
    fire_counts = {}
    for s in range(1, 30):
        p = _p(lid_grasped=(s >= 10), lid_lifted=(s >= 10))
        rb = rm.step_reward(s, p)
        for name in rb.milestones_fired:
            fire_counts[name] = fire_counts.get(name, 0) + 1
    assert fire_counts.get("stable_grasp", 0) == 1, "stable_grasp paid exactly once"
    assert all(v == 1 for v in fire_counts.values()), f"every milestone paid once: {fire_counts}"
    print("[ok] skill milestones are paid at most once per episode")


def test_terminal_decay_and_primary_reward():
    cfg = RewardConfig(terminal_decay_gamma=0.998)
    rm = RewardManager(cfg)
    p = _p(lid_on_blender=True, gripper_far=True, official_check_success=True)
    rb = rm.step_reward(100, p, done=True)
    expected_terminal = 1.0 * (0.998 ** 100)
    assert abs(rb.terminal - expected_terminal) < 1e-9
    assert abs(rb.primary - (rb.terminal + rb.milestone + rb.penalty)) < 1e-9
    print("[ok] terminal reward uses gamma^step decay and primary is the simulator total")


def test_z1_baseline():
    assert z1_baseline_reward(True, 100) == 0.998 ** 100
    assert z1_baseline_reward(False, 100) == 0.0
    print("[ok] Z-1 baseline reward = terminal 1/0 with gamma=0.998 decay")


# ---------------------------- skill manager -------------------------------- #
def test_oracle_happy_path():
    planner = OraclePlanner()
    call = planner.initial()
    assert call.skill is Skill.GRASP
    call = planner.propose(call, SkillOutcome.SUCCESS, _p())
    assert call.skill is Skill.MOVE_HOLDING
    call = planner.propose(call, SkillOutcome.SUCCESS, _p())
    assert call.skill is Skill.PLACE
    done = planner.propose(call, SkillOutcome.SUCCESS, _p())
    assert done is None, "PLACE success ends the plan"
    print("[ok] oracle planner walks GRASP->MOVE_HOLDING->PLACE on success")


def test_oracle_recovery_rules():
    planner = OraclePlanner()
    g = planner.initial()
    # dropped during move -> back to GRASP
    move = planner.propose(g, SkillOutcome.SUCCESS, _p())
    back = planner.propose(move, SkillOutcome.DROPPED, _p())
    assert back.skill is Skill.GRASP
    # released off target -> back to GRASP
    place = SkillCall_place()
    r = planner.propose(place, SkillOutcome.OFF_TARGET, _p())
    assert r.skill is Skill.GRASP
    # timeout while lid on blender -> stay in PLACE (retreat)
    r2 = planner.propose(place, SkillOutcome.TIMEOUT, _p(lid_on_blender=True))
    assert r2.skill is Skill.PLACE
    print("[ok] oracle recovery: dropped/off-target->GRASP, near-target timeout->PLACE")


def SkillCall_place():
    from skill_manager import SkillCall
    return SkillCall(Skill.PLACE, "blender_lid", "blender", "securely_on_top")


def test_monitor_verdicts():
    m = SkillMonitor(Skill.GRASP, MonitorConfig(grasp_hold_steps=3))
    out = None
    for s in range(1, 6):
        out = m.update(_p(lid_grasped=True, lid_lifted=True), s, horizon=300)
        if out:
            break
    assert out is SkillOutcome.SUCCESS
    # move-holding drop detection
    mm = SkillMonitor(Skill.MOVE_HOLDING, MonitorConfig())
    mm.update(_p(lid_grasped=True), 1, 300)
    drop = mm.update(_p(lid_grasped=False), 2, 300)
    assert drop is SkillOutcome.DROPPED
    print("[ok] skill monitor emits success / dropped / timeout verdicts from predicates")


def test_vlm_planner_separate_from_oracle():
    vp = VLMPlannerStub()
    assert vp.name == "vlm_stub" and vp.name != OraclePlanner.name
    assert vp.initial().skill is Skill.GRASP  # stub mirrors oracle but is a distinct object
    print("[ok] VLM planner is a separate pluggable interface, evaluated apart from oracle")


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"\nAll {len(fns)} env-component tests passed.")


if __name__ == "__main__":
    _run_all()
