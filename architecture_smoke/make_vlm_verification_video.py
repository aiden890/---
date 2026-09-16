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

    def flush(verdict, steps, success_step, reason):
        if cur is None:
            return
        cur["verdict"] = verdict
        cur["steps"] = steps
        cur["success_step"] = success_step
        cur["reason"] = reason
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
                  cur.get("_verify_reason"))
            cur = None

    for seg in segments:
        frames = seg["frames"]
        for i, fi in enumerate(frames):
            ann[fi] = {
                "skill": seg["skill"], "instr": seg["instr"], "plan_idx": seg["plan_idx"],
                "verdict": seg["verdict"], "steps": seg["steps"],
                "success_step": seg["success_step"], "reason": seg["reason"],
                "max_prob": seg["max_prob"],
                "vlm": seg["vlm_by_frame"].get(fi),
                "is_boundary": (i == len(frames) - 1),
            }
    if 0 not in ann and segments:
        s0 = segments[0]
        ann[0] = {"skill": s0["skill"], "instr": s0["instr"], "plan_idx": s0["plan_idx"],
                  "verdict": None, "steps": s0["steps"], "success_step": s0["success_step"],
                  "reason": None, "max_prob": s0["max_prob"], "vlm": None, "is_boundary": False}
    return ann, segments


def _place_note(skill, verdict):
    if skill == "PLACE_OBJECT" and verdict != "SUCCESS":
        return "  [single-frame PLACE VQA is weak -> possible false negative]"
    return ""


def draw(frame, a, seed):
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

    # bottom strip: live VLM P(yes) + latch + decision
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
        cv2.putText(frame, f"VLM P(yes)={p:.2f}  latch={vlm.get('consec')}"
                    f"  decision={dec}",
                    (pad, h - 14), font, 0.44, dc, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"executing {a['skill']} ... (VLM not yet queried / CONTINUE)",
                    (pad, h - 14), font, 0.44, (170, 170, 170), 1, cv2.LINE_AA)

    if at_boundary:
        vc = DECISION_COLOR.get("ADVANCE" if a["verdict"] == "SUCCESS" else "REPLAN",
                                (200, 200, 200))
        mp = a.get("max_prob")
        mp_s = f"{mp:.2f}" if mp is not None else "n/a"
        if a["verdict"] == "SUCCESS":
            txt = f"VLM ADVANCE -> SUCCESS (max P(yes)={mp_s})"
        else:
            txt = (f"VLM REPLAN/timeout (max P(yes)={mp_s})"
                   + _place_note(a["skill"], a["verdict"]))
        cv2.putText(frame, txt[:88], (pad, h - 2), font, 0.40, vc, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), vc, 4)
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

    cap = cv2.VideoCapture(os.path.join(seed_dir, "episode.mp4"))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))

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
            frame = draw(frame, a, args.seed)
            reps = args.hold_boundary if a.get("is_boundary") else 1
        else:
            reps = 1
        for _ in range(reps):
            vw.write(frame)
            written += 1
        fi += 1
    cap.release()
    vw.release()
    print(json.dumps({"seed": args.seed, "video_frames_in": fi, "frames_written": written,
                      "segments": [(s["skill"], s["verdict"],
                                    round(s["max_prob"], 3) if s["max_prob"] is not None else None)
                                   for s in segments],
                      "out": args.out}))


if __name__ == "__main__":
    main()
