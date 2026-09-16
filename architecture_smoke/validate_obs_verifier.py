"""Offline validation harness: obs-only verifier vs simulator ground truth.

Replays a completed rollout through the obs-only ``ObsVLMVerifier`` and measures
how well its ADVANCE (skill-done) decision agrees with the simulator's
privileged ground-truth success step -- WITHOUT ever feeding a privileged
predicate to the verifier.

Inputs per rollout (produced by the existing defined-instruction rollouts):
  * ``<skill>.mp4``          : the recorded 3-cam frames, horizontally concat'd
                               (rollout.make_video_frame: L | R | eye_in_hand).
  * ``<skill>_steps.jsonl``  : per-step records with the sim ``predicates`` and
                               ``official_success`` / ``success_step`` -- used
                               ONLY as the offline ground-truth label, never fed
                               to the verifier.

The composed video frame is split back into the 3 camera views and handed to the
verifier as obs. (These older rollouts did not dump the 14-D proprio stream, so
the harness drives the verifier in its VLM-cadence mode -- ``event_gated=False``;
the proprio gate is unit-tested separately.)

Ground-truth success step per skill (privileged, offline label only):
  GRASP  : first step where ``lid_grasped`` has held for the grasp hold window
           (the rollout's own ``success_step`` when it succeeded).
  PLACE  : first step where ``official_check_success`` is true (``success_step``).
  A rollout with no success_step is a ground-truth NEGATIVE (skill never done).

Metrics (aggregated over rollouts, per skill):
  * confusion: verifier ADVANCE vs ground-truth success (TP/FP/FN/TN),
  * precision / recall / F1,
  * timing offset on true positives: verifier_frame_step - gt_success_step
    (negative = verifier fired early, positive = late),
  * overrun: for ground-truth successes the verifier missed (FN) or fired far
    after success.

No success is fabricated: every number is computed from the recorded frames and
the recorded sim labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
#  frame <-> obs                                                                #
# --------------------------------------------------------------------------- #
def split_three_cam(frame):
    """Split an L|R|eye_in_hand horizontal concat back into 3 camera arrays."""
    import numpy as np
    frame = np.asarray(frame)
    w = frame.shape[1]
    if w % 3 != 0:
        raise ValueError(f"composed frame width {w} not divisible by 3")
    cw = w // 3
    from obs_verifier import CAMERA_KEYS
    return {CAMERA_KEYS[i]: np.ascontiguousarray(frame[:, i * cw:(i + 1) * cw])
            for i in range(3)}


def load_frames(mp4_path):
    import imageio.v2 as imageio
    return list(imageio.mimread(mp4_path, memtest=False))


# --------------------------------------------------------------------------- #
#  ground truth (privileged -- offline label only)                              #
# --------------------------------------------------------------------------- #
def load_steps(jsonl_path):
    """Return per-step list of dicts and the recorded frame->step mapping."""
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


def gt_success_step(skill, steps, success):
    """Simulator ground-truth done step for this skill (or None if never)."""
    if success and success.get("success_step") is not None:
        return int(success["success_step"])
    # fall back to scanning predicates
    key = "official_check_success"
    for d in steps:
        if skill.upper().startswith("GRASP"):
            if d["predicates"].get("lid_grasped"):
                # grasp success needs the hold; the rollout encodes it in success
                continue
        if d.get("official_success") or d["predicates"].get(key):
            return int(d["step"])
    return None


# --------------------------------------------------------------------------- #
#  replay one rollout                                                            #
# --------------------------------------------------------------------------- #
def replay(rollout_dir, skill, backend, *, vlm_min_interval, hysteresis_k, tau,
           max_steps):
    from obs_verifier import ObsInput, ObsVLMVerifier

    rollout_dir = Path(rollout_dir)
    mp4 = rollout_dir / f"{skill}.mp4"
    jsonl = rollout_dir / f"{skill}_steps.jsonl"
    frames = load_frames(mp4)
    steps, frame_map, success = load_steps(jsonl)
    gt = gt_success_step(skill, steps, success)

    # frame_map: list of (env_step, frame_index) for the recorded frames. The
    # first recorded frame is the settled start (frame_index 0). Align verifier
    # 'elapsed' to the env step of each recorded frame.
    skill_name = {"grasp": "GRASP_OBJECT", "move": "MOVE_OBJECT",
                  "place": "PLACE_OBJECT"}[skill]
    verifier = ObsVLMVerifier(skill_name, backend, max_steps=max_steps,
                              vlm_min_interval=vlm_min_interval,
                              hysteresis_k=hysteresis_k, tau=tau,
                              event_gated=False)

    advance_frame_step = None
    per_query = []
    # map frame_index -> env_step
    fidx_to_step = {fi: st for (st, fi) in frame_map}
    n = min(len(frames), max(fidx_to_step) + 1 if fidx_to_step else len(frames))
    for fidx in range(n):
        env_step = fidx_to_step.get(fidx, fidx)
        images = split_three_cam(frames[fidx])
        obs = ObsInput(images=images, proprio=None, step=env_step)
        r = verifier.update(obs)
        if verifier.query_log and verifier.query_log[-1]["step"] == verifier.elapsed:
            q = dict(verifier.query_log[-1]); q["env_step"] = env_step
            per_query.append(q)
        if r.decision.value == "ADVANCE":
            advance_frame_step = env_step
            break

    return {
        "rollout": rollout_dir.name, "skill": skill,
        "gt_success_step": gt, "gt_success": gt is not None,
        "obs_advance_step": advance_frame_step,
        "obs_advanced": advance_frame_step is not None,
        "n_vlm_calls": verifier.n_vlm_calls,
        "vlm_seconds": round(verifier.vlm_seconds, 4),
        "mean_vlm_latency_ms": verifier.stats()["mean_vlm_latency_ms"],
        "queries": per_query,
        "n_recorded_frames": len(frames),
    }


# --------------------------------------------------------------------------- #
#  aggregate metrics                                                             #
# --------------------------------------------------------------------------- #
def aggregate(records):
    by_skill = {}
    for r in records:
        by_skill.setdefault(r["skill"], []).append(r)
    out = {}
    all_off = []
    for skill, rs in by_skill.items():
        tp = fp = fn = tn = 0
        offsets = []
        for r in rs:
            gt, adv = r["gt_success"], r["obs_advanced"]
            if gt and adv:
                tp += 1
                offsets.append(r["obs_advance_step"] - r["gt_success_step"])
            elif not gt and adv:
                fp += 1
            elif gt and not adv:
                fn += 1
            else:
                tn += 1
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
        all_off += offsets
        out[skill] = {
            "n": len(rs), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1,
            "timing_offset_steps": offsets,
            "mean_timing_offset": (sum(offsets) / len(offsets)) if offsets else None,
            "median_timing_offset": (sorted(offsets)[len(offsets) // 2]) if offsets else None,
        }
    out["_overall"] = {
        "n": len(records),
        "mean_timing_offset": (sum(all_off) / len(all_off)) if all_off else None,
        "total_vlm_calls": sum(r["n_vlm_calls"] for r in records),
        "mean_vlm_latency_ms": _mean([r["mean_vlm_latency_ms"] for r in records
                                      if r["mean_vlm_latency_ms"] is not None]),
    }
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


# --------------------------------------------------------------------------- #
#  backend factory                                                              #
# --------------------------------------------------------------------------- #
def make_backend(kind, model_path, device, robot_type):
    if kind == "mock":
        # deterministic mock: says yes once past the middle of the stream.
        from vlm_backends import MockVLMBackend
        state = {"i": 0}

        def sched(i, q):
            return 0.9 if i >= 3 else 0.1
        return MockVLMBackend(sched)
    if kind == "qwen":
        import os
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        from vlm_backends import QwenVLMScorerBackend
        if device == "cpu":
            n = int(os.environ.get("OMP_NUM_THREADS", "16"))
            torch.set_num_threads(n)
            torch.set_num_interop_threads(max(1, n // 4))
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                          dtype=dtype).to(device).eval()
        proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return QwenVLMScorerBackend(model, proc, tok, robot_type=robot_type,
                                    device=device, dtype=dtype)
    raise ValueError(f"unknown backend {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts-root", required=True,
                    help="dir containing per-rollout subdirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["mock", "qwen"], default="mock")
    ap.add_argument("--model-path", default="/checkpoint")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--robot-type", default="robocasa365")
    ap.add_argument("--vlm-min-interval", type=int, default=16)
    ap.add_argument("--hysteresis-k", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--glob", default="*",
                    help="subdir glob under rollouts-root (e.g. 'grasp/seed*')")
    ap.add_argument("--skill", choices=["grasp", "move", "place", "auto"],
                    default="auto")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.rollouts_root)

    backend = make_backend(args.backend, args.model_path, args.device, args.robot_type)

    records = []
    dirs = sorted(p for p in root.glob(args.glob) if p.is_dir())
    for d in dirs:
        # detect skill from files present
        if args.skill == "auto":
            skill = None
            for s in ("grasp", "move", "place"):
                if (d / f"{s}_steps.jsonl").exists():
                    skill = s
                    break
            if skill is None:
                continue
        else:
            skill = args.skill
        try:
            r = replay(d, skill, backend, vlm_min_interval=args.vlm_min_interval,
                       hysteresis_k=args.hysteresis_k, tau=args.tau,
                       max_steps=args.max_steps)
        except Exception as e:  # noqa: BLE001
            r = {"rollout": d.name, "skill": skill, "error": repr(e)}
        records.append(r)
        print(json.dumps({k: r.get(k) for k in
                          ("rollout", "skill", "gt_success_step", "obs_advance_step",
                           "n_vlm_calls", "error")}), flush=True)

    good = [r for r in records if "error" not in r]
    metrics = aggregate(good)
    result = {
        "config": vars(args), "n_rollouts": len(records),
        "n_ok": len(good), "metrics": metrics, "records": records,
    }
    (out / "validation.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({"metrics": metrics}, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
