"""make_verification_video.py -- overlay the harness state onto each rollout frame.

Task t_91bfee2b. Renders a *verification* video from an existing smoke-test
episode: for every recorded video frame it draws which skill is executing, the
planner-call index, the rendered instruction, the per-skill step, the verifier
hold counter, and -- at a skill boundary -- the verifier verdict (ADVANCE
=success / TIMEOUT=REPLAN). This makes the architecture's skill boundaries,
instruction switches and verifier decisions visible to the eye, exactly per the
task's video requirement.

It reuses the ALREADY-RENDERED frames in ``seed<N>/episode.mp4`` (produced on the
GPU box) and the ``seed<N>/trace.jsonl`` control record -- no simulator or GPU
needed here. Pure cv2 + the trace.

Frame<->trace mapping (from trace.py + executor.py):
  * video frame 0 is the env-reset frame (before any step).
  * every 'frame' trace record carries {skill, step (per-skill elapsed),
    frame_index}; frame_index counts into the same episode.mp4.
  * a skill segment = the frames between two planner boundaries; the verify
    record at the end gives its verdict.

Run:
  python3 make_verification_video.py --root <architecture_base_smoke dir> \
      --seed 0 --out <path.mp4>
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
VERDICT_COLOR = {"SUCCESS": (80, 220, 120), "TIMEOUT": (60, 60, 235)}
SHORT_INSTR = {
    "GRASP_OBJECT": "Grasp the lid at its handle and lift it clear.",
    "MOVE_OBJECT": "Move the held lid to the pre-place region.",
    "PLACE_OBJECT": "Place the lid on the blender, release, retreat.",
}


def build_frame_annotations(recs):
    """Return {frame_index: annotation dict} for every recorded video frame.

    Walks the trace in order, tracking the active planner boundary / skill /
    verdict so each frame carries the live harness state at capture time.
    """
    ann = {}
    plan_idx = -1
    cur_skill = None
    cur_instr = None
    seg_verdict = None      # verdict of the *segment this frame belongs to*
    # First pass: attach verdict to each segment by scanning results.
    # We stream: plan -> route -> window* -> frame* -> verify -> skill_result.
    # A frame emitted before 'verify' still belongs to the segment whose verdict
    # arrives at that verify; so we buffer frame indices per segment.
    seg_frames = []
    segments = []           # list of (skill, instr, plan_idx, [frame_idx...], verdict)
    for r in recs:
        t = r["type"]
        if t == "plan":
            if r.get("selected_skill") is not None:
                plan_idx = r["planner_calls"]
                cur_skill = r["selected_skill"]
                cur_instr = r["rendered_instruction"]
                seg_frames = []
        elif t == "frame":
            seg_frames.append(r["frame_index"])
        elif t == "skill_result":
            segments.append({"skill": cur_skill, "instr": cur_instr,
                             "plan_idx": plan_idx, "frames": list(seg_frames),
                             "verdict": r["status"], "steps": r["steps"],
                             "success_step": r.get("success_step")})
    # frame 0 = reset, belongs to the first segment as its "pre" frame
    for seg in segments:
        for fi in seg["frames"]:
            ann[fi] = {
                "skill": seg["skill"], "instr": seg["instr"],
                "plan_idx": seg["plan_idx"], "verdict": seg["verdict"],
                "steps": seg["steps"], "success_step": seg["success_step"],
                "is_boundary": fi == seg["frames"][-1] if seg["frames"] else False,
            }
    if 0 not in ann and segments:
        s0 = segments[0]
        ann[0] = {"skill": s0["skill"], "instr": s0["instr"], "plan_idx": s0["plan_idx"],
                  "verdict": None, "steps": s0["steps"], "success_step": s0["success_step"],
                  "is_boundary": False}
    return ann, segments


def draw(frame, a, fi, seed):
    h, w = frame.shape[:2]
    pad = 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = SKILL_COLOR.get(a["skill"], (200, 200, 200))
    at_boundary = a.get("is_boundary") and a["verdict"]
    # top banner strip
    cv2.rectangle(frame, (0, 0), (w, 44), (28, 28, 28), -1)
    cv2.rectangle(frame, (0, 0), (w, 44), color, 2)
    label = f"seed{seed}  planner_call#{a['plan_idx']}  SKILL: {a['skill']}"
    cv2.putText(frame, label, (pad, 18), font, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, SHORT_INSTR.get(a["skill"], a["instr"] or "")[:70],
                (pad, 36), font, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
    # bottom strip: neutral "executing" DURING the skill; verdict only AT the
    # boundary frame -- so the overlay never claims success before it happens.
    cv2.rectangle(frame, (0, h - 26), (w, h), (28, 28, 28), -1)
    if at_boundary:
        vc = VERDICT_COLOR.get(a["verdict"], (200, 200, 200))
        verdict_txt = ("VERIFIER: ADVANCE (done_when latched -> SUCCESS)"
                       if a["verdict"] == "SUCCESS"
                       else "VERIFIER: REPLAN (budget exhausted -> TIMEOUT)")
        cv2.putText(frame, verdict_txt, (pad, h - 8), font, 0.44, vc, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), vc, 4)  # emphasise boundary
    else:
        cv2.putText(frame, f"executing {a['skill']} ... (verifier: CONTINUE)",
                    (pad, h - 8), font, 0.44, (170, 170, 170), 1, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "defined-instruction-rollouts", "architecture_base_smoke"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--hold-boundary", type=int, default=15,
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
            frame = draw(frame, a, fi, args.seed)
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
                      "segments": [(s["skill"], s["verdict"], s["steps"]) for s in segments],
                      "out": args.out}))


if __name__ == "__main__":
    main()
