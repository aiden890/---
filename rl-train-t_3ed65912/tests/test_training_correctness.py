from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from training_correctness import (  # noqa: E402
    ExactHoldWindow,
    RewardComponents,
    append_progress,
    guarded_ratio,
    hold_enabled_for_variant,
    mean_loss_scale,
    nonduplicated_skill_reward,
    reset_gated_store,
    skill_completion_reward,
    skill_timeout_reward,
    source_manifest,
    should_stop_for_kl,
    validate_optimizer_config,
    verify_deployment_manifest,
    verify_source_manifest,
)
from update_batch import validate_batch_config  # noqa: E402

_LOCAL_REWARD = Path(__file__).resolve().parents[2] / "rl-env-t_4f3f2b20" / "src" / "reward.py"
_REWARD_PATH = _LOCAL_REWARD if _LOCAL_REWARD.exists() else Path("/rl_env/src/reward.py")
_reward_spec = importlib.util.spec_from_file_location("canonical_reward", _REWARD_PATH)
_reward = importlib.util.module_from_spec(_reward_spec)
sys.modules[_reward_spec.name] = _reward
_reward_spec.loader.exec_module(_reward)


def test_completion_reward_decays_and_can_be_disabled():
    assert skill_completion_reward(0, 1.0, 0.998, True) == 1.0
    assert math.isclose(skill_completion_reward(100, 1.0, 0.998, True), 0.998**100)
    assert skill_completion_reward(100, 1.0, 0.998, False) == 1.0
    assert skill_completion_reward(None, 1.0, 0.998, True) == 0.0


def test_official_terminal_prevents_duplicate_skill_payment():
    assert nonduplicated_skill_reward(1.0, 12, 1.0, 0.9, True) == 0.0
    assert math.isclose(nonduplicated_skill_reward(0.0, 12, 1.0, 0.9, True), 0.9**12)


def test_skill_timeout_penalty_applies_only_to_timeout():
    assert skill_timeout_reward("TIMEOUT", 0.5) == -0.5
    assert skill_timeout_reward("SUCCESS", 0.5) == 0.0


def test_terminal_only_ablation_disables_hold_shaping():
    assert not hold_enabled_for_variant("simulator_terminal_only", 20)
    assert hold_enabled_for_variant("terminal_plus_hold", 20)
    assert not hold_enabled_for_variant("terminal_plus_hold", 0)


def test_gated_single_group_resets_store_but_deferred_batch_group_does_not():
    class Client:
        resets = 0
        def reset_store(self):
            self.resets += 1

    client = Client()
    assert reset_gated_store(client, defer_update=False)
    assert client.resets == 1
    assert not reset_gated_store(client, defer_update=True)
    assert client.resets == 1


def test_reward_components_sum_and_stay_separate():
    r = RewardComponents(skill_terminal=0.8, official_terminal=0.2, hold_stay=0.1,
                         hold_drift=-0.05, hold_lapse=-0.5, drop=-0.25,
                         collision=-0.05, timeout=-0.1)
    assert math.isclose(r.total(), 0.15)
    data = r.as_dict()
    assert {"skill_terminal", "official_terminal", "hold_stay", "hold_drift",
            "hold_lapse", "drop", "collision", "timeout", "total"}.issubset(data)
    assert math.isclose(data["total"], sum(data[k] for k in data if k != "total"))


def test_environment_penalty_breakdown_names_each_cause():
    cfg = _reward.RewardConfig(settle_terminal=False)
    mgr = _reward.RewardManager(cfg)
    mgr.step_reward(1, {"lid_grasped": True})
    rb = mgr.step_reward(2, {
        "lid_grasped": False, "lid_on_blender": False,
        "gripper_lid_contact": False, "lid_other_contacts": True,
    }, truncated=True)
    assert rb.object_dropped == cfg.penalties["object_dropped"]
    assert rb.disallowed_collision == 0.0
    assert rb.timeout == cfg.penalties["timeout"]
    assert rb.penalty == rb.object_dropped + rb.disallowed_collision + rb.timeout


def test_exact_hold_window_finishes_after_exactly_n_post_success_steps_despite_lapse():
    h = ExactHoldWindow(3)
    assert not h.observe(success_now=False)
    assert not h.observe(success_now=True)  # latch; not a post-success step
    assert not h.observe(success_now=False)
    assert not h.observe(success_now=True)
    assert h.observe(success_now=False)
    assert h.held_steps == 3
    assert h.lapse_steps == 2


def test_exact_hold_window_zero_finishes_on_success():
    h = ExactHoldWindow(0)
    assert not h.observe(False)
    assert h.observe(True)
    assert h.held_steps == 0


def test_exact_hold_window_stops_early_only_for_environment_termination():
    h = ExactHoldWindow(20)
    assert not h.observe(True)
    assert h.observe(True, terminated=True)
    assert h.held_steps == 1


def test_batch_config_rejects_invalid_bounds():
    assert validate_batch_config(8, 1024, 24) is None
    for bad in [(0, 1, 1), (2, -1, 3), (4, 1, 3)]:
        try:
            validate_batch_config(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid batch config accepted: {bad}")


def test_optimizer_config_rejects_inert_or_unsafe_values():
    assert validate_optimizer_config(2, 10.0, 0.2, 1.0, 0.1) == (2, 10.0, 0.2, 1.0, 0.1)
    for bad in [(0, 10, .2, 1, .1), (1, 1, .2, 1, .1),
                (1, 10, 0, 1, .1), (1, 10, .2, 0, .1), (1, 10, .2, 1, -1)]:
        try:
            validate_optimizer_config(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid optimizer config accepted: {bad}")


def test_gradient_normalization_is_batch_size_invariant():
    single_gradient = 4.0
    for repeats in (1, 7, 1024):
        accumulated = sum(single_gradient * mean_loss_scale(repeats) for _ in range(repeats))
        assert math.isclose(accumulated, single_gradient)


def test_raw_ratio_guard_rejects_before_clamp():
    assert math.isclose(guarded_ratio(0.0, 10.0), 1.0)
    assert guarded_ratio(math.log(10.0) + 1e-4, 10.0) is None
    assert guarded_ratio(-math.log(10.0) - 1e-4, 10.0) is None
    assert guarded_ratio(float("nan"), 10.0) is None


def test_target_kl_never_stops_epoch0_and_stops_later():
    assert not should_stop_for_kl(0, 1.0, 0.02)
    assert not should_stop_for_kl(1, 0.01, 0.02)
    assert should_stop_for_kl(1, 0.03, 0.02)
    assert not should_stop_for_kl(3, 999.0, None)


def test_progress_jsonl_is_append_only_and_parseable():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "group_progress.jsonl"
        append_progress(p, {"update": 0, "group": 0, "stored_chunks": 4})
        append_progress(p, {"update": 0, "group": 1, "stored_chunks": 9})
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        assert [r["group"] for r in rows] == [0, 1]


def test_source_manifest_detects_mutation():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "a.py").write_text("x=1\n")
        expected = source_manifest(root, ["a.py"])
        verify_source_manifest(root, expected)
        (root / "a.py").write_text("x=2\n")
        try:
            verify_source_manifest(root, expected)
        except RuntimeError as e:
            assert "a.py" in str(e)
        else:
            raise AssertionError("source hash mutation escaped gate")


def test_deployment_manifest_returns_commit_and_rejects_dirty_marker():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "a.py").write_text("x=1\n")
        manifest = {"commit": "abc123", "dirty": False,
                    "files": source_manifest(root, ["a.py"])}
        path = root / "source_manifest.json"
        path.write_text(json.dumps(manifest))
        assert verify_deployment_manifest(root, path)["commit"] == "abc123"
        manifest["dirty"] = True
        path.write_text(json.dumps(manifest))
        try:
            verify_deployment_manifest(root, path)
        except RuntimeError as e:
            assert "dirty" in str(e)
        else:
            raise AssertionError("dirty deployment manifest escaped gate")


def test_mutations_flip_clean_invariants():
    clean = RewardComponents(skill_terminal=1.0).as_dict()
    mutated = copy.deepcopy(clean)
    mutated["total"] += 1.0
    component_sum = sum(mutated[k] for k in mutated if k != "total")
    assert not math.isclose(mutated["total"], component_sum)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("training correctness tests passed")
