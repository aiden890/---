"""Annotate corpus PLACE frames with boundary-judge vs strict-success-gate vs
sim GT (task t_32a4f9b6), to make the false-positive fix legible to a human.

Uses the ALREADY-recorded corpus frames (<skill>.mp4) + the cached per-view
P(yes) (cache_perview.json). No GPU, no simulator. For each frame it draws:
  * boundary judge (OLD single-Q latch): P(yes) + FIRE->SUCCESS,
  * strict success gate (NEW): calibrated final-frame combined@eye checkpoint,
  * sim GT (offline label): success step / never.
A seed the OLD boundary judge calls SUCCESS while the strict gate + sim GT say
FAILURE (the PLACE false positive this task fixes) is explicit on screen.

PIL + imageio only (the xiaomi-cu121 image has both; no cv2 needed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def load_cache_rollout(cache, skill, seed):
    for ro in cache["rollouts"]:
        if ro["skill"] == skill and ro["seed"] == seed:
            return ro
    raise SystemExit(f"no cached rollout {skill}/seed{seed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--skill", default="place")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boundary-tau", type=float, default=0.98)
    ap.add_argument("--boundary-k", type=int, default=2)
    args = ap.parse_args()

    cache = json.loads(Path(args.cache).read_text())
    ro = load_cache_rollout(cache, args.skill, args.seed)
    gt_step, gt_succ = ro["gt_success_step"], ro["gt_success"]

    gate_key, gate_thr = "place_combined@eye", 0.899728536605835

    mp4 = Path(args.corpus) / args.skill / f"seed{args.seed}" / f"{args.skill}.mp4"
    frames = list(imageio.mimread(mp4, memtest=False))
    by_fidx = {f["frame_index"]: f for f in ro["frames"]}

    pad = 150
    out_frames = []
    boundary_run, boundary_fired = 0, False
    n_obs = 0

    def col(c):
        return c

    for fidx, raw in enumerate(frames):
        arr = np.asarray(raw)[:, :, :3]
        H, W = arr.shape[:2]
        canvas = np.zeros((H + pad, W, 3), dtype=np.uint8)
        canvas[:] = (24, 20, 20)
        canvas[:H, :, :] = arr
        img = Image.fromarray(canvas)
        d = ImageDraw.Draw(img)
        rec = by_fidx.get(fidx)
        if rec is not None:
            sc = rec["scores"]
            bkey = gate_key
            bp = sc.get(bkey, 0.0)
            if bp >= args.boundary_tau:
                boundary_run += 1
                if boundary_run >= args.boundary_k:
                    boundary_fired = True
            else:
                boundary_run = 0
            n_obs += 1
            endpoint_p = sc.get(gate_key, 0.0)
            gate_pass = endpoint_p >= gate_thr and n_obs >= 5

            y = H + 8
            d.text((10, y), f"{args.skill.upper()} seed{args.seed}  frame {fidx}",
                   fill=(230, 230, 230))
            d.text((10, y + 26),
                   f"BOUNDARY (old): P(yes)={bp:.2f}  "
                   f"{'FIRE->SUCCESS' if boundary_fired else 'waiting'}",
                   fill=(120, 220, 80) if boundary_fired else (170, 170, 170))
            d.text((10, y + 52),
                   f"SUCCESS GATE (new endpoint): combined@eye={endpoint_p:.2f}>={gate_thr:.2f}  "
                   f"-> {'PASS' if gate_pass else 'REJECT'}",
                   fill=(120, 220, 80) if gate_pass else (240, 130, 60))
            d.text((10, y + 78),
                   f"SIM GT: success@{gt_step}" if gt_succ else "SIM GT: FAILURE (never)",
                   fill=(120, 220, 80) if gt_succ else (240, 130, 60))
            if boundary_fired and not gt_succ:
                d.text((10, y + 104),
                       "<< OLD FALSE POSITIVE (boundary=SUCCESS, sim=FAIL); new gate REJECTS",
                       fill=(240, 130, 60))
        out_frames.append(np.asarray(img))

    imageio.mimwrite(args.out, out_frames, fps=20, quality=7)
    print(f"wrote {args.out} ({len(out_frames)} frames)")


if __name__ == "__main__":
    main()
