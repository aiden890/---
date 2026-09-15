"""Real VLM-auxiliary reward scorer for CloseBlenderLid (task t_37303cd3).

The policy's own frozen backbone is a genuine **Qwen3-VL-4B-Instruct** vision-language
model with a full language-model head (config.json: vlm_config._name_or_path =
"Qwen/Qwen3-VL-4B-Instruct"). We reuse THAT model as a real vision success/progress
scorer -- no hand-coded predicate, no separate CLIP download: we pose a yes/no visual
question about the scene and read the model's answer likelihood from its logits.

    score = P("yes" | image, question)
          = softmax over the {yes, no} answer tokens at the first answer position.

This is a true vision-model judgement of the frame: it looks at the rendered camera
image and the natural-language question ("is the blender lid closed and resting on the
blender base?") and returns how confident the VLM is that the answer is yes. It is used
ONLY as the auxiliary channel in reward.py (vlm_weight>0); the simulator predicate stays
the primary reward and the arbiter of success. This module is imported by BOTH the
GRPO trainer server (which owns the GPU model and runs the forward) and the training
client (which owns the processor and builds the inputs), so the VQA prompt and the
logit->probability math live in exactly one place (single source of truth).

Split of work (no model duplication):
  client (has processor)  : build_vqa_inputs(processor, image, question) -> cpu tensors
  server (has GPU model)  : run model.vlm(**inputs).logits, then answer_probability(...)

Questions are task milestones so the same scorer serves both the gate (official success)
and the training-time shaping (grasp progress):
"""
from __future__ import annotations

from typing import Any, Optional

# ---- VQA question bank (CloseBlenderLid). Each asks a single yes/no visual fact. ----
QUESTIONS = {
    # official terminal success -- used by the verification gate
    "success": (
        "These are camera views of a robot manipulation scene showing a blender. Has the "
        "blender lid been placed back on top of the blender base so the blender is closed, "
        "AND has the robot gripper let go of the lid and moved away from it? "
        "Answer yes or no."
    ),
    # grasp progress -- used as the training-time auxiliary shaping for the GRASP skill
    "grasp": (
        "These are camera views of a robot manipulation scene. Is the robot gripper firmly "
        "grasping and holding the blender lid right now? Answer yes or no."
    ),
    # generic progress toward closing the blender
    "progress": (
        "These are camera views of a robot manipulation scene. Is the robot making progress "
        "toward closing the blender by putting its lid back on top? Answer yes or no."
    ),
}
DEFAULT_QUESTION = "grasp"

# yes/no surface forms; the first that tokenizes to a single id wins (resolved lazily).
_YES_FORMS = ("Yes", " Yes", "yes", " yes")
_NO_FORMS = ("No", " No", "no", " no")


def resolve_yes_no_ids(tokenizer) -> tuple[list[int], list[int]]:
    """All single-token ids for yes / no surface forms (robust to tokenizer quirks)."""
    def ids_for(forms):
        out = []
        for f in forms:
            toks = tokenizer.encode(f, add_special_tokens=False)
            if len(toks) == 1:
                out.append(toks[0])
        return sorted(set(out))
    yes = ids_for(_YES_FORMS)
    no = ids_for(_NO_FORMS)
    if not yes or not no:
        raise RuntimeError(f"could not resolve yes/no token ids (yes={yes} no={no})")
    return yes, no


def build_vqa_prompt(question: str) -> str:
    """A minimal Qwen3-VL chat prompt: one image then the yes/no question."""
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_vqa_inputs(processor, image, question_key: str = DEFAULT_QUESTION,
                     *, robot_type: str, state_dim: int = 60, state_length: int = 4):
    """Build cpu VLM inputs for one image + yes/no question (client side).

    Returns a plain dict of cpu tensors: input_ids, attention_mask, pixel_values,
    image_grid_thw. `state`/`action_mask` are attached by the processor but unused by the
    VLM forward, so they are dropped here to keep the payload small.
    """
    import numpy as np
    from PIL import Image
    import torch

    q = QUESTIONS.get(question_key, question_key)
    prompt = build_vqa_prompt(q)
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    dummy_state = torch.zeros((1, state_length, state_dim), dtype=torch.float32)
    feat = processor(images=[image], text=[prompt], state=dummy_state,
                     robot_type=robot_type, return_tensors="pt")
    keep = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    return {k: feat[k] for k in keep if k in feat}


def answer_probability(logits_last, yes_ids, no_ids) -> float:
    """P(yes) from the first-answer-position logits, restricted to the yes/no tokens.

    logits_last: 1-D tensor over the vocabulary at the position that predicts the answer.
    """
    import torch
    lg = logits_last.float()
    yes_logit = torch.logsumexp(lg[yes_ids], dim=0)
    no_logit = torch.logsumexp(lg[no_ids], dim=0)
    two = torch.stack([yes_logit, no_logit])
    p = torch.softmax(two, dim=0)[0]
    return float(p)


class VLMScorerClient:
    """Client-side helper: builds VQA inputs and RPCs the trainer server's op=vlm_score.

    `rpc` is a callable(request_dict) -> response_dict (e.g. TrainerClient._rpc). The
    server returns the scalar probability directly, so the same prompt/math module runs
    on both ends and no vocabulary-sized tensor crosses the wire.
    """

    def __init__(self, processor, rpc, robot_type: str,
                 state_dim: int = 60, state_length: int = 4):
        self.processor = processor
        self.rpc = rpc
        self.robot_type = robot_type
        self.state_dim = state_dim
        self.state_length = state_length

    def score(self, image, question_key: str = DEFAULT_QUESTION) -> float:
        inputs = build_vqa_inputs(self.processor, image, question_key,
                                  robot_type=self.robot_type,
                                  state_dim=self.state_dim, state_length=self.state_length)
        resp = self.rpc({"op": "vlm_score", "inputs": inputs, "question": question_key})
        return float(resp["prob"])
