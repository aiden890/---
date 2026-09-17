"""CPU tests for portable short-sequence boundary inference."""
from sequence_boundary import SequenceBoundaryModel, sequence_features
from obs_verifier import CAMERA_KEYS, ObsInput, ObsVLMVerifier, VLMBackend
from schemas import Decision


def spec(*, bias=10.0, dwell=2, view="full"):
    return {
        "feature_mean": [0.0] * 72,
        "feature_scale": [1.0] * 72,
        "layers": [{"weights": [[0.0] for _ in range(72)], "bias": [bias]}],
        "tau": 0.9,
        "dwell": dwell,
        "view": view,
    }


def proprio(x=0.0):
    return [x] * 14


def images():
    return {key: [[0]] for key in CAMERA_KEYS}


class Backend(VLMBackend):
    def score(self, images, question_text):
        return 0.01


def test_feature_contract_is_causal_and_72d():
    short = sequence_features([0.2], [proprio(0.0)], 1)
    longer = sequence_features([0.2, 0.8], [proprio(0.0), proprio(0.1)], 2)
    assert len(short) == len(longer) == 72
    assert short[0] == 0.2 and longer[0] == 0.8
    assert abs(longer[4] - 0.6) < 1e-12


def test_explicit_post_positive_dwell():
    model = SequenceBoundaryModel(spec(dwell=2))
    first, p1 = model.update(0.1, proprio(), 1)
    second, p2 = model.update(0.1, proprio(), 2)
    assert p1 > 0.9 and p2 > 0.9
    assert first is False and second is True


def test_portable_tree_inference():
    tree_spec = spec(dwell=1)
    tree_spec.pop("layers")
    tree_spec["trees"] = [{
        "feature": [0, -2, -2], "threshold": [0.5, -2.0, -2.0],
        "left": [1, -1, -1], "right": [2, -1, -1],
        "probability": [0.0, 0.1, 0.95],
    }]
    model = SequenceBoundaryModel(tree_spec)
    assert model.update(0.2, proprio(), 1)[0] is False
    assert model.update(0.8, proprio(), 2)[0] is True


def test_obs_verifier_uses_sequence_model_instead_of_raw_vlm_threshold():
    verifier = ObsVLMVerifier("GRASP_OBJECT", Backend(), max_steps=10,
                              vlm_min_interval=1, event_gated=False,
                              tau=0.99, hysteresis_k=9,
                              sequence_model=spec(dwell=2))
    first = verifier.update(ObsInput(images=images(), proprio=proprio()))
    second = verifier.update(ObsInput(images=images(), proprio=proprio()))
    assert first.decision == Decision.CONTINUE
    assert second.decision == Decision.ADVANCE
    assert "sequence verifier" in second.reason


def test_sequence_view_mismatch_is_rejected():
    try:
        ObsVLMVerifier("GRASP_OBJECT", Backend(), max_steps=10, view="eye",
                       sequence_model=spec(view="left"))
    except ValueError as exc:
        assert "expects view" in str(exc)
    else:
        raise AssertionError("view mismatch was accepted")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print("[PASS]", test.__name__)
    print(f"{len(tests)}/{len(tests)} checks passed")
