"""Focused real-VLM discrimination probe (task t_d4268a2e).

A full per-step stream sweep of a 4B VLM is impractical on a contended CPU
(~6 min/forward measured) while the GPU is owned by the parallel training task.
This probe instead spends a small, fixed budget of real VLM forwards where they
carry the most signal: it scores frames at *known ground-truth phases* with the
matching skill-completion question and checks that the VLM's P(yes) SEPARATES
"skill done" from "skill not done" -- the core question behind the obs-only
verifier ("can a VLM judge skill completion from obs alone?").

For each rollout it picks, per skill, up to 3 labeled frames using ONLY the
recorded sim ground-truth (offline supervision label -- never fed to a runtime
verifier):
  * NEG_early : an early frame, well before the skill's ground-truth success
                (expected P(yes) low).
  * POS       : the ground-truth success frame (expected P(yes) high) -- only
                for rollouts that actually succeeded.
  * for a rollout that never succeeded: two NEG frames (early + late) -- both
                expected low; a high P(yes) here is a false positive.

Outputs a per-frame table (rollout, skill, phase, gt_label, p_yes) plus a
separation summary (mean P(yes) on POS vs NEG, and the implied ROC-style
threshold behaviour). No success is fabricated -- every label is the recorded
sim predicate and every P(yes) is a real forward.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SKILL_NAME = {"grasp": "GRASP_OBJECT", "move": "MOVE_OBJECT", "place": "PLACE_OBJECT"}


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
    return None


def pick_frames(skill, steps, frames, success):
    """Return list of (frame_index, env_step, phase, gt_label_done)."""
    gt = gt_success_step(steps, success)
    fmap = frames  # list of (env_step, frame_index)
    if not fmap:
        return []
    steps_sorted = fmap
    picks = []
    if gt is not None:
        # POS: recorded frame closest to (>=) gt success step
        pos = min(steps_sorted, key=lambda sf: abs(sf[0] - gt))
        # NEG_early: a frame at ~20% of the way to success
        early_step = max(1, int(gt * 0.25))
        neg = min(steps_sorted, key=lambda sf: abs(sf[0] - early_step))
        picks.append((neg[1], neg[0], "neg_early", False))
        picks.append((pos[1], pos[0], "pos_success", True))
    else:
        # two negatives: ~40% and ~90% through the (failed) rollout
        last = steps_sorted[-1][0]
        for frac, tag in ((0.4, "neg_mid"), (0.9, "neg_late")):
            s = max(1, int(last * frac))
            f = min(steps_sorted, key=lambda sf: abs(sf[0] - s))
            picks.append((f[1], f[0], tag, False))
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", default="/checkpoint")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--robot-type", default="robocasa365")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--specs", required=True,
                    help="comma list of subdir:skill, e.g. "
                         "'rand10_grasp_seed0:grasp,rand10_place_seed9:place'")
    args = ap.parse_args()

    import torch
    import imageio.v2 as imageio
    torch.set_num_threads(args.threads)
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    from vlm_backends import QwenVLMScorerBackend, compose_three_cam
    from obs_verifier import CAMERA_KEYS, SKILL_QUESTIONS

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(args.rollouts_root)

    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    t0 = time.time()
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True,
                                      dtype=dtype).to(args.device).eval()
    proc = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    backend = QwenVLMScorerBackend(model, proc, tok, robot_type=args.robot_type,
                                   device=args.device, dtype=dtype)
    load_s = round(time.time() - t0, 1)
    print(json.dumps({"model_loaded_s": load_s}), flush=True)

    def split(frame):
        import numpy as np
        w = frame.shape[1] // 3
        return {CAMERA_KEYS[i]: np.ascontiguousarray(frame[:, i * w:(i + 1) * w])
                for i in range(3)}

    rows = []
    for spec in args.specs.split(","):
        spec = spec.strip()
        if not spec:
            continue
        sub, skill = spec.split(":")
        d = root / sub
        steps, frames, success = load_steps(d / f"{skill}_steps.jsonl")
        vid = imageio.mimread(d / f"{skill}.mp4", memtest=False)
        q = SKILL_QUESTIONS[SKILL_NAME[skill]]
        for (fidx, env_step, phase, gt_done) in pick_frames(skill, steps, frames, success):
            if fidx >= len(vid):
                continue
            imgs = split(vid[fidx])
            t = time.time()
            p = float(backend.score(imgs, q))
            dt = round(time.time() - t, 2)
            row = {"rollout": sub, "skill": skill, "phase": phase,
                   "env_step": env_step, "frame_index": fidx,
                   "gt_done": gt_done, "p_yes": round(p, 4), "forward_s": dt}
            rows.append(row)
            print(json.dumps(row), flush=True)

    # separation summary
    pos = [r["p_yes"] for r in rows if r["gt_done"]]
    neg = [r["p_yes"] for r in rows if not r["gt_done"]]
    def mean(xs): return round(sum(xs) / len(xs), 4) if xs else None
    # best-threshold accuracy over the observed p_yes values
    best = None
    cand = sorted({r["p_yes"] for r in rows})
    for th in cand:
        tp = sum(1 for r in rows if r["gt_done"] and r["p_yes"] >= th)
        tn = sum(1 for r in rows if not r["gt_done"] and r["p_yes"] < th)
        acc = (tp + tn) / len(rows) if rows else 0
        if best is None or acc > best["acc"]:
            best = {"tau": th, "acc": round(acc, 3), "tp": tp, "tn": tn}
    summary = {
        "n_frames": len(rows), "n_pos": len(pos), "n_neg": len(neg),
        "mean_p_yes_pos": mean(pos), "mean_p_yes_neg": mean(neg),
        "separation": (mean(pos) - mean(neg)) if (pos and neg) else None,
        "best_threshold": best,
        "mean_forward_s": mean([r["forward_s"] for r in rows]),
        "model_loaded_s": load_s, "device": args.device, "threads": args.threads,
    }
    result = {"config": vars(args), "summary": summary, "rows": rows}
    (out / "discrimination.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({"summary": summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
