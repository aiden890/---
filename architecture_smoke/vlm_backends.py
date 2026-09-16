"""VLM backends for the obs-only verifier.

``MockVLMBackend``   -- deterministic, no torch/model; drives the unit tests and
                        the offline control-loop check.
``QwenVLMScorerBackend`` -- the real judge. It reuses the policy's own frozen
                        Qwen3-VL backbone through the ``vlm_scorer`` module
                        (single source of truth for the VQA prompt + the
                        logit->probability math). The 3 camera views are
                        composed into one wide image (the same horizontal
                        concat as ``rollout.make_video_frame``) and posed as a
                        single-image yes/no VQA, so no second copy of the
                        prompt/logit code is introduced.

Both honour the same ``VLMBackend.score(images, question_text) -> P(yes)``
interface. Neither ever sees a simulator predicate.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

from obs_verifier import CAMERA_KEYS, VLMBackend


# --------------------------------------------------------------------------- #
#  Mock backend (tests / offline loop)                                          #
# --------------------------------------------------------------------------- #
class MockVLMBackend(VLMBackend):
    """Returns scripted P(yes) values, ignoring the pixels.

    ``schedule`` is a callable(call_index, question_text) -> float, or a plain
    list of floats consumed in order (last value repeats). Lets a test make the
    VLM "recognise completion" at a chosen query index and exercise hysteresis.
    """

    def __init__(self, schedule):
        if callable(schedule):
            self._fn = schedule
            self._seq = None
        else:
            self._seq = list(schedule)
            self._fn = None
        self.calls = 0
        self.seen_questions: list[str] = []

    def score(self, images: Mapping[str, Any], question_text: str) -> float:
        # still exercise the obs contract: images must be present, no predicates.
        if not images:
            raise ValueError("MockVLMBackend got empty images")
        self.seen_questions.append(question_text)
        i = self.calls
        self.calls += 1
        if self._fn is not None:
            return float(self._fn(i, question_text))
        if not self._seq:
            return 0.0
        return float(self._seq[min(i, len(self._seq) - 1)])


# --------------------------------------------------------------------------- #
#  Camera composition                                                           #
# --------------------------------------------------------------------------- #
def compose_three_cam(images: Mapping[str, Any]):
    """Horizontally concatenate the 3 camera views into one wide RGB image.

    Identical composition to ``rollout.make_video_frame`` (axis=1 concat of
    agentview_left | agentview_right | eye_in_hand), so the VLM sees every view
    the policy sees in a single forward.
    """
    import numpy as np

    def get(key):
        for cand in (key, "video." + key):
            if cand in images:
                return np.asarray(images[cand], dtype=np.uint8)
        raise KeyError(f"camera {key!r} missing (have {sorted(images)})")

    return np.ascontiguousarray(np.concatenate([get(k) for k in CAMERA_KEYS], axis=1))


# --------------------------------------------------------------------------- #
#  Real Qwen3-VL backend                                                        #
# --------------------------------------------------------------------------- #
class QwenVLMScorerBackend(VLMBackend):
    """Real VQA success judge on the policy's own Qwen3-VL backbone.

    ``model`` must expose ``.vlm(input_ids, attention_mask, pixel_values,
    image_grid_thw).logits`` (the Xiaomi RoboCasa365 MiBot model does). ``proc``
    is the matching processor. ``compose`` turns the 3-cam dict into one image
    (defaults to the horizontal 3-cam concat). yes/no token ids are resolved
    once from the model's tokenizer via ``vlm_scorer.resolve_yes_no_ids``.

    The prompt + probability math come from ``vlm_scorer`` unchanged.
    """

    def __init__(self, model, proc, tokenizer, *, robot_type: str,
                 device: str = "cpu", dtype=None,
                 state_dim: int = 60, state_length: int = 4,
                 compose: Callable[[Mapping[str, Any]], Any] = compose_three_cam):
        import vlm_scorer
        self._vs = vlm_scorer
        self.model = model
        self.proc = proc
        self.robot_type = robot_type
        self.device = device
        self.dtype = dtype
        self.state_dim = state_dim
        self.state_length = state_length
        self.compose = compose
        self._yes_ids, self._no_ids = vlm_scorer.resolve_yes_no_ids(tokenizer)

    def score(self, images: Mapping[str, Any], question_text: str) -> float:
        import torch

        image = self.compose(images)
        inputs = self._vs.build_vqa_inputs(
            self.proc, image, question_text, robot_type=self.robot_type,
            state_dim=self.state_dim, state_length=self.state_length)
        dev_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point() and self.dtype is not None:
                    v = v.to(self.dtype)
                v = v.to(self.device)
            dev_inputs[k] = v
        with torch.no_grad():
            out = self.model.vlm(
                input_ids=dev_inputs["input_ids"],
                attention_mask=dev_inputs.get("attention_mask"),
                pixel_values=dev_inputs.get("pixel_values"),
                image_grid_thw=dev_inputs.get("image_grid_thw"),
            )
            logits_last = out.logits[0, -1, :]
            return self._vs.answer_probability(logits_last, self._yes_ids, self._no_ids)
