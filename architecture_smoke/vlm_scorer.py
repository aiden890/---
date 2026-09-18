"""Minimal VQA utilities for the observation-only skill termination verifier.

This module is intentionally isolated from RL reward computation. It reads only rendered
camera images and provides yes/no probabilities from the policy's frozen Qwen3-VL
backbone. Simulator predicates remain the sole training reward source.
"""
from __future__ import annotations

QUESTIONS = {
    "success": (
        "These are camera views of a robot manipulation scene showing a blender. Has the "
        "blender lid been placed back on top of the blender base so the blender is closed, "
        "AND has the robot gripper let go of the lid and moved away from it? Answer yes or no."
    ),
    "grasp": (
        "These are camera views of a robot manipulation scene. Is the robot gripper firmly "
        "grasping and holding the blender lid right now? Answer yes or no."
    ),
    "progress": (
        "These are camera views of a robot manipulation scene. Is the robot making progress "
        "toward closing the blender by putting its lid back on top? Answer yes or no."
    ),
}
DEFAULT_QUESTION = "grasp"
_YES_FORMS = ("Yes", " Yes", "yes", " yes")
_NO_FORMS = ("No", " No", "no", " no")


def resolve_yes_no_ids(tokenizer) -> tuple[list[int], list[int]]:
    def ids_for(forms):
        ids = []
        for form in forms:
            tokens = tokenizer.encode(form, add_special_tokens=False)
            if len(tokens) == 1:
                ids.append(tokens[0])
        return sorted(set(ids))

    yes_ids, no_ids = ids_for(_YES_FORMS), ids_for(_NO_FORMS)
    if not yes_ids or not no_ids:
        raise RuntimeError(f"could not resolve yes/no token ids (yes={yes_ids} no={no_ids})")
    return yes_ids, no_ids


def build_vqa_prompt(question: str) -> str:
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_vqa_inputs(
    processor,
    image,
    question_key: str = DEFAULT_QUESTION,
    *,
    robot_type: str,
    state_dim: int = 60,
    state_length: int = 4,
):
    import numpy as np
    from PIL import Image
    import torch

    question = QUESTIONS.get(question_key, question_key)
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    dummy_state = torch.zeros((1, state_length, state_dim), dtype=torch.float32)
    features = processor(
        images=[image],
        text=[build_vqa_prompt(question)],
        state=dummy_state,
        robot_type=robot_type,
        return_tensors="pt",
    )
    keep = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    return {key: features[key] for key in keep if key in features}


def answer_probability(logits_last, yes_ids, no_ids) -> float:
    import torch

    logits = logits_last.float()
    yes_logit = torch.logsumexp(logits[yes_ids], dim=0)
    no_logit = torch.logsumexp(logits[no_ids], dim=0)
    return float(torch.softmax(torch.stack([yes_logit, no_logit]), dim=0)[0])
