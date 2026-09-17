"""Tests for deterministic balanced calibration-corpus selection."""
import json
import tempfile
from pathlib import Path

from build_balanced_corpus import build_balanced_corpus


def test_selects_exact_balanced_success_and_failure_per_skill():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src, dst = root / "src", root / "dst"
        labels = {
            0: {"grasp": True, "move_holding": False},  # legacy: no proprio, excluded
            1: {"grasp": False, "move_holding": True},
            2: {"grasp": True, "move_holding": True},
            3: {"grasp": False, "move_holding": False},
            4: {"grasp": True, "move_holding": False},
        }
        for seed, row in labels.items():
            sd = src / f"seed{seed}"
            sd.mkdir(parents=True)
            results = {}
            for skill, success in row.items():
                if success is not None:
                    results[skill] = {"success": success}
                    (sd / f"{skill}.mp4").write_bytes(b"video")
                    step = {} if seed == 0 else {"proprio": [0.0] * 14}
                    (sd / f"{skill}_steps.jsonl").write_text(json.dumps(step) + "\n")
            (sd / "results.json").write_text(json.dumps(results))

        manifest = build_balanced_corpus(
            src, dst, skills=("grasp", "move_holding"), positives=2, negatives=2)

        assert manifest["skills"]["grasp"]["positive_seeds"] == [2, 4]
        assert manifest["skills"]["grasp"]["negative_seeds"] == [1, 3]
        assert manifest["skills"]["move_holding"]["positive_seeds"] == [1, 2]
        assert manifest["skills"]["move_holding"]["negative_seeds"] == [3, 4]
        assert (dst / "grasp" / "seed2").is_symlink()
        assert json.loads((dst / "manifest.json").read_text()) == manifest


def test_raises_when_a_class_is_under_target():
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "src", Path(td) / "dst"
        sd = src / "seed0"
        sd.mkdir(parents=True)
        (sd / "results.json").write_text(json.dumps({"grasp": {"success": True}}))
        (sd / "grasp_steps.jsonl").write_text(json.dumps({"proprio": [0.0] * 14}) + "\n")
        try:
            build_balanced_corpus(src, dst, skills=("grasp",), positives=1, negatives=1)
        except ValueError as exc:
            assert "grasp" in str(exc) and "negative" in str(exc)
        else:
            raise AssertionError("under-target class must fail")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"{len(tests)}/{len(tests)} checks passed")
