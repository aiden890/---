"""make_vlm_verification_video.py -- overlay the OBS-ONLY VLM verifier state onto
each rollout frame (task t_fc5e73d5).

Renders a verification video from a ``run_vlm_inference.py`` episode: for every
recorded frame it draws which skill is executing, the rendered instruction, and
the obs-only Qwen3-VL verifier's judgement at that moment -- the VQA question,
the latest P(yes), the consecutive-yes latch count, and the decision
(CONTINUE / ADVANCE=SUCCESS / REPLAN=timeout). GRASP / MOVE / PLACE are colour-
coded and named so the skill boundaries are legible.

HONEST LABELLING (operator requirement): the overlay reports the VLM's actual
P(yes) and does NOT dress a weak judgement as success. In particular PLACE is a
known weak single-frame VQA case (a true success can read low P(yes) = a false
negative); when a PLACE skill ends without a VLM latch the overlay says
"REPLAN (VLM P(yes) below tau -- known single-frame PLACE weakness)" rather than
implying the placement failed. A per-skill confidence chip shows the max P(yes)
the VLM reached for that skill.

Reuses the ALREADY-RENDERED ``seed<N>/episode.mp4`` frames + ``trace.jsonl``
(the 'vlm' + 'verify' records). Pure cv2 + the trace: no simulator, no GPU.

Run:
  python3 make_vlm_verification_video.py --root <out dir> --seed 0 --out <path.mp4>
"""
from __future__ import annotations

import argparse
import json
import os

import cv2

SKILL_COLOR = {
    "GRASP_OBJECT": (80, 200, 120),   # green  (BGR)
    "MOVE_OBJECT": (230, 160, 40),    # blue
    "PLACE_OBJECT": (60, 130, 240),   # orange
}
DECISION_COLOR = {
    "ADVANCE": (80, 220, 120),        # green
    "CONTINUE": (170, 170, 170),      # grey
    "REPLAN": (60, 60, 235),          # red
    "TERMINATED": (60, 60, 235),
}
SHORT_INSTR = {
    "GRASP_OBJECT": "Grasp the lid at its handle and lift it clear.",
    "MOVE_OBJECT": "Move the held lid above the blender (pre-place).",
    "PLACE_OBJECT": "Place lid on blender, release, retreat.",
}
SHORT_Q = {
    "GRASP_OBJECT": "VLM asks: is the gripper holding the lid, lifted clear?",
    "MOVE_OBJECT": "VLM asks: is the lid held above the blender, ready to place?",
    "PLACE_OBJECT": "VLM asks: is the lid back on the blender AND gripper let go?",
}


def build_frame_annotations(recs):
    """{frame_index: annotation} for every recorded frame, from the VLM trace.

    Streams: plan -> route -> window* -> (frame / vlm)* -> verify -> skill_result.
    Each 'vlm' record carries {skill, step, frame_index, prob, consecutive_yes,
    queried, decision}. We attach to each frame the LAST VLM judgement at/<= it,
    plus the segment's final verdict + per-skill max P(yes).
    """
    ann = {}
    segments = []
    cur = None
    last_vlm = None   # last vlm record in this segment

    def flush(verdict, steps, success_step, reason, success_gate=None):
        if cur is None:
            return
        cur["verdict"] = verdict
        cur["steps"] = steps
        cur["success_step"] = success_step
        cur["reason"] = reason
        cur["success_gate"] = success_gate   # strict, view-routed, episode-level
        segments.append(cur)

    for r in recs:
        t = r["type"]
        if t == "plan":
            if r.get("selected_skill") is not None:
                cur = {"skill": r["selected_skill"], "instr": r["rendered_instruction"],
                       "plan_idx": r["planner_calls"], "frames": [], "vlm_by_frame": {},
                       "max_prob": None, "verdict": None, "steps": None,
                       "success_step": None, "reason": None}
                last_vlm = None
        elif t == "frame" and cur is not None:
            fi = r["frame_index"]
            cur["frames"].append(fi)
            # carry forward the last VLM judgement to this frame
            cur["vlm_by_frame"][fi] = last_vlm
        elif t == "vlm" and cur is not None:
            last_vlm = {"prob": r.get("prob"), "consec": r.get("consecutive_yes"),
                        "queried": r.get("queried"), "decision": r.get("decision")}
            fi = r.get("frame_index")
            if fi is not None:
                cur["vlm_by_frame"][fi] = last_vlm
            p = r.get("prob")
            if p is not None and (cur["max_prob"] is None or p > cur["max_prob"]):
                cur["max_prob"] = p
        elif t == "verify" and cur is not None:
            # verify closes the segment (decision = ADVANCE/REPLAN/TERMINATED)
            cur["_verify_decision"] = r.get("decision")
            cur["_verify_reason"] = r.get("reason")
        elif t == "skill_result" and cur is not None:
            flush(r["status"], r["steps"], r.get("success_step"),
                  cur.get("_verify_reason"), r.get("success_gate"))
            cur = None

    for seg in segments:
        frames = seg["frames"]
        for i, fi in enumerate(frames):
            ann[fi] = {
                "skill": seg["skill"], "instr": seg["instr"], "plan_idx": seg["plan_idx"],
                "verdict": seg["verdict"], "steps": seg["steps"],
                "success_step": seg["success_step"], "reason": seg["reason"],
                "max_prob": seg["max_prob"], "success_gate": seg.get("success_gate"),
                "vlm": seg["vlm_by_frame"].get(fi),
                "is_boundary": (i == len(frames) - 1),
            }
    if 0 not in ann and segments:
        s0 = segments[0]
        ann[0] = {"skill": s0["skill"], "instr": s0["instr"], "plan_idx": s0["plan_idx"],
                  "verdict": None, "steps": s0["steps"], "success_step": s0["success_step"],
                  "reason": None, "max_prob": s0["max_prob"], "success_gate": s0.get("success_gate"),
                  "vlm": None, "is_boundary": False}
    return ann, segments


GREEN = (80, 220, 120)
RED = (60, 60, 235)
GREY = (170, 170, 170)
AMBER = (40, 190, 235)


def _gate_summary(sg):
    """Compact one-line strict-gate read: PASS/FAIL + per-sub view scores."""
    if not sg:
        return None, None
    ok = bool(sg.get("success"))
    scores = sg.get("sub_scores") or []
    views = sg.get("sub_views") or []
    thr = sg.get("sub_thresholds") or []
    parts = []
    for s, v, t in zip(scores, views, thr):
        s_s = f"{s:.2f}" if isinstance(s, (int, float)) else "n/a"
        parts.append(f"{v}={s_s}/{t:.2f}")
    detail = " ".join(parts) if parts else (sg.get("reason") or "")
    return ok, detail


def draw(frame, a, seed, sim_task_success=None):
    h, w = frame.shape[:2]
    pad = 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = SKILL_COLOR.get(a["skill"], (200, 200, 200))
    at_boundary = a.get("is_boundary") and a["verdict"]

    # top banner: skill + instruction
    cv2.rectangle(frame, (0, 0), (w, 60), (28, 28, 28), -1)
    cv2.rectangle(frame, (0, 0), (w, 60), color, 2)
    cv2.putText(frame, f"seed{seed}  plan#{a['plan_idx']}  SKILL: {a['skill']}",
                (pad, 17), font, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, SHORT_INSTR.get(a["skill"], (a["instr"] or ""))[:74],
                (pad, 34), font, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, SHORT_Q.get(a["skill"], "")[:74],
                (pad, 52), font, 0.40, (150, 190, 230), 1, cv2.LINE_AA)

    # right chip: obs-only judge tag
    cv2.putText(frame, "judge: obs VLM (Qwen3-VL)", (w - 210, 17), font, 0.40,
                (140, 200, 250), 1, cv2.LINE_AA)

    # bottom strip: live VLM P(yes) + latch + decision (BOUNDARY judge)
    cv2.rectangle(frame, (0, h - 46), (w, h), (28, 28, 28), -1)
    vlm = a.get("vlm")
    if vlm and vlm.get("prob") is not None:
        p = vlm["prob"]
        dec = vlm.get("decision", "CONTINUE")
        dc = DECISION_COLOR.get(dec, (170, 170, 170))
        # P(yes) bar
        bar_w = int((w - 2 * pad) * max(0.0, min(1.0, p)))
        cv2.rectangle(frame, (pad, h - 40), (w - pad, h - 30), (60, 60, 60), -1)
        cv2.rectangle(frame, (pad, h - 40), (pad + bar_w, h - 30), dc, -1)
        cv2.putText(frame, f"BOUNDARY(latch) P(yes)={p:.2f}  latch={vlm.get('consec')}"
                    f"  decision={dec}",
                    (pad, h - 14), font, 0.44, dc, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"executing {a['skill']} ... (VLM not yet queried / CONTINUE)",
                    (pad, h - 14), font, 0.44, (170, 170, 170), 1, cv2.LINE_AA)

    if at_boundary:
        # ---- TWO SEPARATE JUDGMENTS at the skill boundary -----------------
        #  (1) BOUNDARY judge (latch): did it ADVANCE / timeout?
        #  (2) STRICT SuccessGate (view-routed, episode-level): PASS/FAIL.
        # Showing them apart is the whole point: a PLACE that ADVANCEd on a
        # transient single-frame flush now visibly FAILs the strict gate.
        adv = a["verdict"] == "SUCCESS"
        bc = GREEN if adv else RED
        mp = a.get("max_prob")
        mp_s = f"{mp:.2f}" if mp is not None else "n/a"
        boundary_txt = (f"[1] BOUNDARY: {'ADVANCE' if adv else 'REPLAN/timeout'}"
                        f" (max P(yes)={mp_s})")

        gate_ok, gate_detail = _gate_summary(a.get("success_gate"))
        # panel geometry: draw an opaque box above the bottom strip
        y0 = h - 46 - 62
        cv2.rectangle(frame, (0, y0), (w, h - 46), (20, 20, 20), -1)
        cv2.putText(frame, boundary_txt[:78], (pad, y0 + 16), font, 0.44, bc, 1, cv2.LINE_AA)

        if gate_ok is None:
            # skills without a strict gate (MOVE) -- say so, don't fake a verdict
            cv2.putText(frame, "[2] SUCCESS GATE: n/a (waypoint skill, no strict gate)",
                        (pad, y0 + 36), font, 0.42, GREY, 1, cv2.LINE_AA)
            outer = bc
        else:
            gc = GREEN if gate_ok else RED
            gtxt = f"[2] SUCCESS GATE(strict): {'PASS' if gate_ok else 'FAIL'}  {gate_detail}"
            cv2.putText(frame, gtxt[:82], (pad, y0 + 36), font, 0.42, gc, 1, cv2.LINE_AA)
            # highlight the caught false-positive: latch ADVANCE but strict FAIL
            if adv and not gate_ok:
                cv2.putText(frame,
                            "  --> ADVANCE but STRICT=FAIL: PLACE false-positive CAUGHT",
                            (pad, y0 + 54), font, 0.42, AMBER, 1, cv2.LINE_AA)
            outer = gc  # the strict verdict drives the frame border

        # ---- sim GT chip (offline label ONLY, never a verifier input) ------
        if sim_task_success is not None and a["skill"] == "PLACE_OBJECT":
            sg_c = GREEN if sim_task_success else RED
            sim_txt = f"sim GT (offline): task_success={sim_task_success}"
            (tw, _), _ = cv2.getTextSize(sim_txt, font, 0.42, 1)
            cv2.putText(frame, sim_txt, (w - tw - pad, y0 + 16), font, 0.42, sg_c, 1, cv2.LINE_AA)
            # agreement note between the strict obs gate and the sim label
            if gate_ok is not None:
                agree = (gate_ok == bool(sim_task_success))
                atxt = "strict==sim (obs gate agrees w/ GT)" if agree else "strict!=sim (disagree)"
                ac = GREEN if agree else AMBER
                (aw, _), _ = cv2.getTextSize(atxt, font, 0.40, 1)
                cv2.putText(frame, atxt, (w - aw - pad, y0 + 36), font, 0.40, ac, 1, cv2.LINE_AA)

        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), outer, 4)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing seed<N>/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--hold-boundary", type=int, default=18,
                    help="repeat each skill-boundary frame N times so the verdict is readable")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    seed_dir = os.path.join(root, f"seed{args.seed}")
    recs = [json.loads(l) for l in open(os.path.join(seed_dir, "trace.jsonl"))]
    ann, segments = build_frame_annotations(recs)

    # sim GT (offline label ONLY -- never a verifier input). Read the seed
    # summary's official predicate so the overlay can show VLM strict-success
    # vs actual sim success side by side.
    sim_task_success = None
    try:
        with open(os.path.join(seed_dir, "summary.json")) as f:
            sim_task_success = bool(json.load(f).get("task_success"))
    except Exception:
        sim_task_success = None

    cap = cv2.VideoCapture(os.path.join(seed_dir, "episode.mp4"))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # h264 via imageio-ffmpeg so the mp4 plays in browsers (cv2 lacks an h264 encoder
    # in the client image -> FMP4/mp4v which browsers refuse). Buffer RGB frames.
    import imageio.v2 as imageio
    out_frames = []

    fi = 0
    written = 0
    last_ann = ann.get(0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        a = ann.get(fi, last_ann)
        if a is not None:
            last_ann = a
            frame = draw(frame, a, args.seed, sim_task_success=sim_task_success)
            reps = args.hold_boundary if a.get("is_boundary") else 1
        else:
            reps = 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for _ in range(reps):
            out_frames.append(rgb)
            written += 1
        fi += 1
    cap.release()
    imageio.mimsave(args.out, out_frames, fps=args.fps, codec="libx264",
                    macro_block_size=None, pixelformat="yuv420p")
    print(json.dumps({"seed": args.seed, "video_frames_in": fi, "frames_written": written,
                      "sim_task_success": sim_task_success,
                      "segments": [(s["skill"], s["verdict"],
                                    round(s["max_prob"], 3) if s["max_prob"] is not None else None,
                                    (s.get("success_gate") or {}).get("success"))
                                   for s in segments],
                      "out": args.out}))


if __name__ == "__main__":
    main()
