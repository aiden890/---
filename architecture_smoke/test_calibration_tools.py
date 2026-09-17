"""Tests for data-driven multi-skill calibration tooling."""
import cache_frame_scores
import calibrate_obs_verifier


def test_view_argument_is_parsed_as_names_not_characters():
    assert calibrate_obs_verifier.parse_views("left,right,eye") == ["left", "right", "eye"]


def test_binary_auc_handles_perfect_inverted_and_tied_scores():
    auc = calibrate_obs_verifier.roc_auc_binary
    assert auc([False, False, True, True], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc([False, True], [0.9, 0.1]) == 0.0
    assert auc([False, True], [0.5, 0.5]) == 0.5


def test_move_question_is_cached_for_move_rollouts():
    assert cache_frame_scores.SKILL_QUESTIONS["move_holding"] == ["move"]
    assert "above the blender" in cache_frame_scores.QUESTION_BANK["move"]


def test_candidate_rules_cover_every_requested_view_for_move():
    rules = calibrate_obs_verifier.candidate_rules("move_holding", ["left", "right", "eye"])
    assert set(rules) == {"move@left", "move@right", "move@eye"}
    sample = {
        "move@left": 0.1,
        "move@right": 0.2,
        "move@eye": 0.3,
    }
    assert rules["move@left"](sample) == 0.1
    assert rules["move@right"](sample) == 0.2
    assert rules["move@eye"](sample) == 0.3


def test_candidate_rules_cover_every_requested_view_for_existing_skills():
    views = ["left", "right", "eye"]
    grasp = calibrate_obs_verifier.candidate_rules("grasp", views)
    place = calibrate_obs_verifier.candidate_rules("place", views)
    assert set(grasp) == {"grasp@left", "grasp@right", "grasp@eye"}
    assert "place_combined@right" in place
    assert "place_seated_and_clear_min@eye" in place


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"{len(tests)}/{len(tests)} checks passed")
