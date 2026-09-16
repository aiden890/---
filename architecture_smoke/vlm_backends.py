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

import pickle
import socket
import struct

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


# --------------------------------------------------------------------------- #
#  Remote Qwen3-VL backend (RPCs the combined infer+verify GPU server)          #
# --------------------------------------------------------------------------- #
class RemoteVLMScorerBackend(VLMBackend):
    """Obs-only VLM verifier backend that RPCs ``op=vlm_score`` to the GPU server.

    The client (CPU sim container) owns the processor and builds the VQA inputs
    with the env card's ``vlm_scorer.build_vqa_inputs`` (single source of truth
    for the prompt + tokenisation); the server (GPU) runs ``model.vlm(...)`` and
    returns just the scalar P(yes). The 3-cam dict is composed into one wide RGB
    image exactly as ``rollout.make_video_frame`` (horizontal concat), so the VLM
    sees every view the policy sees. No simulator predicate is ever referenced.

    This backend holds its OWN persistent socket to the same server the action
    client uses (the server threads + a CUDA lock serialise the two request
    types), so no second model load is needed.
    """

    def __init__(self, processor, host, port, *, robot_type: str,
                 state_dim: int = 60, state_length: int = 4,
                 compose: Callable[[Mapping[str, Any]], Any] = None):
        import vlm_scorer
        self._vs = vlm_scorer
        self.processor = processor
        self.host = host
        self.port = port
        self.robot_type = robot_type
        self.state_dim = state_dim
        self.state_length = state_length
        self.compose = compose or compose_three_cam
        self._sock = None
        self._connect()

    def _connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        self._sock = s

    def _rpc(self, req: dict) -> dict:
        payload = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
        self._sock.sendall(struct.pack(">I", len(payload)) + payload)
        ln = self._recv_all(4)
        n = struct.unpack(">I", ln)[0]
        return pickle.loads(self._recv_all(n))

    def _recv_all(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            pkt = self._sock.recv(n - len(buf))
            if not pkt:
                raise ConnectionError("verifier socket closed mid-response")
            buf += pkt
        return buf

    def score(self, images: Mapping[str, Any], question_text: str) -> float:
        image = self.compose(images)
        # build_vqa_inputs accepts a literal question string (falls back to it
        # when the key is not in the QUESTIONS bank), so the obs_verifier's
        # per-skill question text is used verbatim.
        inputs = self._vs.build_vqa_inputs(
            self.processor, image, question_text, robot_type=self.robot_type,
            state_dim=self.state_dim, state_length=self.state_length)
        resp = self._rpc({"op": "vlm_score", "inputs": inputs, "question": question_text})
        return float(resp["prob"])

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
