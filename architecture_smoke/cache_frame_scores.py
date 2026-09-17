"""GPU frame-score caching pass for obs-verifier calibration (task t_32a4f9b6).

This is the ONLY GPU work in the calibration loop. It runs the frozen Qwen3-VL
backbone VQA forward over EVERY recorded frame of the calibration corpus, once
per candidate question (and optionally once per single camera view), and writes
the resulting P(yes) scalars to a JSON cache. Every downstream tau / latch /
hold_steps / question-decomposition / viewpoint experiment then replays that
cache on the CPU for free -- no model reload, no second GPU pass.

Corpus = ``defined-instruction-rollouts/random_g10_p10`` (10 grasp + 10 place
seeds). Each ``<skill>/seed*/`` holds:
  * ``<skill>.mp4``          : recorded 3-cam frames (L | R | eye_in_hand concat)
  * ``<skill>_steps.jsonl``  : per-step sim ``predicates`` + ``official_success``
                               / ``success_step`` -- OFFLINE ground-truth label
                               ONLY, never a verifier input.

The mp4 frame is ALREADY the horizontal 3-cam concat the policy/verifier sees,
so full-concat scoring uses the raw frame directly; per-view scoring splits it
into thirds (obs_verifier.CAMERA_KEYS order).

HARD INVARIANT (obs-only): the VLM sees pixels + a natural-language question
only. Sim predicates are read here solely to attach the offline GT label to the
cache; they never enter build_vqa_inputs.

Output cache schema (``cache.json``):
  {
    "meta": {"model": ..., "robot_type": ..., "questions": {key: text},
             "views": [...], "corpus": ...},
    "rollouts": [
      {"skill": "place", "seed": 0, "rollout": "place/seed0",
       "gt_success": bool, "gt_success_step": int|null, "n_frames": N,
       "frames": [{"frame_index": i, "env_step": s,
                   "scores": {"<qkey>@<view>": prob, ...},
                   "predicates_subset": {...}}, ...]}
    ]
  }
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


# --------------------------------------------------------------------------- #
#  Candidate question bank (obs-only; no privileged pose referenced)            #
# --------------------------------------------------------------------------- #
# GRASP question already scores ~0.98 correct -> keep it as the single grasp Q.
# PLACE is decomposed into sub-facts that AND together, plus the original single
# question for a before/after baseline. Each is a single yes/no visual fact.
QUESTION_BANK = {
    # ---- grasp (works well; kept) ----
    "grasp": (
        "These are camera views of a robot manipulation scene. Is the robot "
        "gripper firmly grasping and holding the blender lid, having lifted it "
        "clear off the counter? Answer yes or no."
    ),
    # ---- move: grasped lid is held above the target, ready to place ----
    "move": (
        "These are camera views of a robot manipulation scene. Is the robot "
        "still holding the blender lid above the blender base, positioned and "
        "ready to place it down? Answer yes or no."
    ),
    # ---- place: original single combined question (baseline for before/after) ----
    "place_combined": (
        "These are camera views of a robot manipulation scene showing a "
        "blender. Has the blender lid been placed back on top of the blender "
        "base so the blender is closed, AND has the robot gripper let go of the "
        "lid and moved away from it? Answer yes or no."
    ),
    # ---- place: decomposed sub-facts (AND-combined offline) ----
    "place_seated": (
        "These are camera views of a robot manipulation scene showing a "
        "blender. Is the blender lid resting flush and fully seated on top of "
        "the blender base, so the blender is properly closed? Answer yes or no."
    ),
    "place_released": (
        "These are camera views of a robot manipulation scene. Has the robot "
        "gripper opened its fingers and let go of the blender lid? "
        "Answer yes or no."
    ),
    "place_clear": (
        "These are camera views of a robot manipulation scene. Is the robot "
        "gripper empty and moved away, no longer touching or holding the "
        "blender lid? Answer yes or no."
    ),
}

# Which questions to score for each skill (avoid wasted forwards).
SKILL_QUESTIONS = {
    "grasp": ["grasp"],
    "move_holding": ["move"],
    "place": ["place_combined", "place_seated", "place_released", "place_clear"],
}

# Predicate keys worth keeping in the cache for offline GT diagnostics (small).
KEEP_PREDICATES = (
    "lid_grasped", "lid_lifted", "lid_on_blender", "in_preplace_region",
    "official_check_success", "lid_upright_7deg", "gripper_lid_far_0.15",
    "gripper_lid_contact", "eef_lid_dist",
)


def split_three_cam(frame):
    import numpy as np
    frame = np.asarray(frame)
    w = frame.shape[1]
    if w % 3 != 0:
        raise ValueError(f"composed frame width {w} not divisible by 3")
    cw = w // 3
    from obs_verifier import CAMERA_KEYS
    return {CAMERA_KEYS[i]: np.ascontiguousarray(frame[:, i * cw:(i + 1) * cw])
            for i in range(3)}


def load_steps(jsonl_path):
    steps, frames, success = [], [], None
    for line in open(jsonl_path):
        d = json.loads(line)
        t = d.get("type")
        if t == "step":
            steps.append(d)
        elif t == "frame":
            frames.append((d["step"], d["frame_index"]))
        elif t == "success":
            success = d
    return steps, frames, success


def gt_success_step(steps, success):
    if success and success.get("success_step") is not None:
        return int(success["success_step"])
    # results.json fallback handled by caller; scan predicates for official.
    for d in steps:
        if d.get("official_success") or d.get("predicates", {}).get("official_check_success"):
            return int(d["step"])
    return None


def build_model(model_path, device):
    import torch
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    kw = dict(trust_remote_code=True, dtype=dtype)
    if device != "cpu":
        try:
            kw["attn_implementation"] = "flash_attention_2"
        except Exception:
            pass
    model = AutoModel.from_pretrained(model_path, **kw)
    model = model.to(device).to(dtype).eval()
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, proc, tok, dtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True,
                    help="root with grasp/seed*/ and place/seed*/")
    ap.add_argument("--out", required=True, help="output cache.json path")
    ap.add_argument("--model", default="/checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--robot-type", default="robocasa365")
    ap.add_argument("--views", default="full",
                    help="comma list of views to score: full,left,right,eye")
    ap.add_argument("--skills", default="grasp,place")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="score every Nth recorded frame (1 = all)")
    ap.add_argument("--peak-check-seed", action="store_true",
                    help="print GPU mem after the first rollout and continue")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np
    import imageio.v2 as imageio
    import torch
    import vlm_scorer
    from obs_verifier import CAMERA_KEYS

    views = args.views.split(",")
    view_to_cam = {"left": CAMERA_KEYS[0], "right": CAMERA_KEYS[1],
                   "eye": CAMERA_KEYS[2]}

    model, proc, tok, dtype = build_model(args.model, args.device)
    yes_ids, no_ids = vlm_scorer.resolve_yes_no_ids(tok)
    print("model loaded", flush=True)

    def score_image(image_arr, question_text):
        inputs = vlm_scorer.build_vqa_inputs(
            proc, image_arr, question_text, robot_type=args.robot_type)
        dev = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    v = v.to(dtype)
                v = v.to(args.device)
            dev[k] = v
        with torch.no_grad():
            out = model.vlm(input_ids=dev["input_ids"],
                            attention_mask=dev.get("attention_mask"),
                            pixel_values=dev.get("pixel_values"),
                            image_grid_thw=dev.get("image_grid_thw"))
            return float(vlm_scorer.answer_probability(out.logits[0, -1, :],
                                                       yes_ids, no_ids))

    corpus = Path(args.corpus)
    rollouts_out = []
    t_start = time.time()
    n_forward = 0

    for skill in args.skills.split(","):
        qkeys = SKILL_QUESTIONS[skill]
        seed_dirs = sorted((corpus / skill).glob("seed*"))
        for sd in seed_dirs:
            seed = int(sd.name.replace("seed", ""))
            mp4 = sd / f"{skill}.mp4"
            jsonl = sd / f"{skill}_steps.jsonl"
            if not mp4.exists() or not jsonl.exists():
                print(f"skip {sd} (missing files)", flush=True)
                continue
            frames = list(imageio.mimread(mp4, memtest=False))
            steps, frame_map, success = load_steps(jsonl)
            gt = gt_success_step(steps, success)
            step_by_frame = {fi: st for (st, fi) in frame_map}
            pred_by_step = {d["step"]: d.get("predicates", {}) for d in steps}
            official_by_step = {d["step"]: bool(d.get("official_success"))
                                for d in steps}

            frame_records = []
            for fidx in range(0, len(frames), args.frame_stride):
                raw = np.asarray(frames[fidx])
                env_step = step_by_frame.get(fidx, fidx)
                scores = {}
                for qk in qkeys:
                    qtext = QUESTION_BANK[qk]
                    for view in views:
                        if view == "full":
                            img = raw
                        else:
                            cams = split_three_cam(raw)
                            img = cams[view_to_cam[view]]
                        scores[f"{qk}@{view}"] = score_image(img, qtext)
                        n_forward += 1
                pred = pred_by_step.get(env_step, {})
                keep = {k: pred.get(k) for k in KEEP_PREDICATES if k in pred}
                keep["official_success"] = official_by_step.get(env_step, False)
                frame_records.append({
                    "frame_index": fidx, "env_step": env_step,
                    "scores": scores, "predicates_subset": keep,
                })
            rollouts_out.append({
                "skill": skill, "seed": seed, "rollout": f"{skill}/{sd.name}",
                "gt_success": gt is not None, "gt_success_step": gt,
                "n_frames": len(frames), "n_scored": len(frame_records),
                "frames": frame_records,
            })
            dt = time.time() - t_start
            print(f"[{skill}/{sd.name}] scored={len(frame_records)} "
                  f"gt_success={gt is not None} gt_step={gt} "
                  f"forwards={n_forward} elapsed={dt:.1f}s", flush=True)
            if args.peak_check_seed and args.device != "cpu":
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / 1e9
                resv = torch.cuda.max_memory_reserved() / 1e9
                print(f"  GPU peak_alloc={peak:.2f}GB peak_reserved={resv:.2f}GB",
                      flush=True)

    out = {
        "meta": {
            "model": args.model, "robot_type": args.robot_type,
            "questions": {k: QUESTION_BANK[k] for sk in args.skills.split(",")
                          for k in SKILL_QUESTIONS[sk]},
            "views": views, "corpus": str(corpus),
            "frame_stride": args.frame_stride,
            "total_forwards": n_forward,
            "elapsed_s": round(time.time() - t_start, 1),
        },
        "rollouts": rollouts_out,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}: {len(rollouts_out)} rollouts, "
          f"{n_forward} forwards, {time.time() - t_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
