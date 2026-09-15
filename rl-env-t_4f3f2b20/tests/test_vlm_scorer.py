"""Unit tests for the real VLM-auxiliary scorer math (task t_37303cd3).

Covers the pure logic that does NOT need the GPU model: yes/no token resolution against
a fake tokenizer, the P(yes) softmax over answer tokens, and the prompt builder. The
end-to-end VQA forward + positive/negative separation is verified on v4 by the
`--vlm-gate-n` gate (results/vlm_gate/vlm_gate.json), which is the real hardware gate.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import vlm_scorer  # noqa: E402


class FakeTok:
    """Minimal tokenizer: 'Yes'/' Yes' -> single id, 'yes' multi-token, etc."""
    _single = {"Yes": 9454, " Yes": 7414, "No": 2822, " No": 1400}

    def encode(self, s, add_special_tokens=False):
        if s in self._single:
            return [self._single[s]]
        return [1, 2]  # multi-token surface form -> rejected by resolver


def test_resolve_yes_no_ids():
    yes, no = vlm_scorer.resolve_yes_no_ids(FakeTok())
    assert set(yes) == {9454, 7414}
    assert set(no) == {2822, 1400}


def test_answer_probability_symmetry():
    vocab = 200
    lg = torch.full((vocab,), -10.0)
    yes_ids, no_ids = [9], [2]
    lg[9] = 5.0; lg[2] = 5.0
    p = vlm_scorer.answer_probability(lg, yes_ids, no_ids)
    assert abs(p - 0.5) < 1e-4, p


def test_answer_probability_confident_yes():
    vocab = 200
    lg = torch.full((vocab,), -10.0)
    lg[9] = 8.0; lg[2] = -2.0
    p = vlm_scorer.answer_probability(lg, [9], [2])
    assert p > 0.99, p


def test_answer_probability_confident_no():
    vocab = 200
    lg = torch.full((vocab,), -10.0)
    lg[9] = -3.0; lg[2] = 7.0
    p = vlm_scorer.answer_probability(lg, [9], [2])
    assert p < 0.01, p


def test_answer_probability_logsumexp_over_multiple_forms():
    # two yes forms should aggregate (logsumexp), not be dropped
    vocab = 200
    lg = torch.full((vocab,), -10.0)
    lg[9] = 3.0; lg[10] = 3.0; lg[2] = 3.0
    p = vlm_scorer.answer_probability(lg, [9, 10], [2])
    assert p > 0.5, p  # two yes tokens outweigh one no at equal logit


def test_prompt_has_single_image_and_question():
    q = "Is the lid closed?"
    prompt = vlm_scorer.build_vqa_prompt(q)
    assert prompt.count("<|image_pad|>") == 1
    assert q in prompt
    assert prompt.rstrip().endswith("assistant")


def test_question_bank_keys():
    assert set(vlm_scorer.QUESTIONS) >= {"success", "grasp", "progress"}
    for v in vlm_scorer.QUESTIONS.values():
        assert "yes or no" in v.lower()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn(); passed += 1; print(f"PASS {fn.__name__}")
    print(f"=== {passed}/{len(fns)} vlm_scorer unit tests passed ===")
