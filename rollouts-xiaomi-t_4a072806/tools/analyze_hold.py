"""Post-success HOLD analysis for skill_eval --post-success-hold runs (t_53b09e39).

For each <skill>_steps.jsonl, split the timeline at ``success_step`` and quantify what the
policy does AFTER the skill first succeeded while the SAME instruction keeps being issued:

  - eef movement per step (mm): mean / max over the post-success window, and total
    displacement from the success frame to the last frame.
  - predicate stability: does the success condition keep holding after success?
    grasp -> lid_grasped ; move_holding -> lid_grasped & in_preplace_region ;
    place -> official_check_success (and lid_on_blender).
  - drift toward the next skill (overfitting signal): lid vertical rise after a grasp,
    lid xy toward closed pos after a move, lid picked back up / knocked after a place.

Verdict per skill (task-defined):
  total post-success eef move < 20mm AND success predicate stays True for >=90% of the
  post window  -> BOUNDARY-RESPECTED (good: policy stops after the command is fulfilled).
  otherwise    -> KEEPS-MOVING / overfitting (policy runs on into the next motion).

Read-only; writes hold_analysis.json + prints a table. Usage: python analyze_hold.py <dir> [<dir> ...]
"""
import json
import math
import sys
from pathlib import Path

SUCCESS_PRED = {
    "grasp": lambda p: p["lid_grasped"],
    "move_holding": lambda p: p["lid_grasped"] and p["in_preplace_region"],
    "place": lambda p: p["official_check_success"],
}


def load_steps(path):
    return [json.loads(l) for l in open(path) if l.startswith('{"type": "step"')]


def dist_mm(a, b):
    return 1000.0 * math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def analyze_one(steps, skill):
    succ = next((r for r in steps if r.get("success_step")), None)
    ss = succ["success_step"] if succ else None
    out = {"skill": skill, "n_steps": len(steps), "success_step": ss}
    if ss is None:
        out["verdict"] = "NO_SUCCESS"
        return out
    post = [r for r in steps if r["step"] >= ss]  # includes the success step itself
    if len(post) < 2:
        out["verdict"] = "NO_HOLD_WINDOW (success at/after horizon)"
        out["post_window_steps"] = len(post) - 1
        return out
    pred = SUCCESS_PRED[skill]
    # per-step eef movement inside the post window
    eef = [r["predicates"]["eef_pos"] for r in post]
    lid = [r["predicates"]["lid_pos"] for r in post]
    step_moves = [dist_mm(eef[i], eef[i - 1]) for i in range(1, len(eef))]
    lid_moves = [dist_mm(lid[i], lid[i - 1]) for i in range(1, len(lid))]
    pred_hold = [bool(pred(r["predicates"])) for r in post]
    frac_hold = sum(pred_hold) / len(pred_hold)
    out.update({
        "post_window_steps": len(post) - 1,
        "eef_move_per_step_mm": {"mean": round(sum(step_moves) / len(step_moves), 2),
                                 "max": round(max(step_moves), 2)},
        "eef_total_move_success_to_end_mm": round(dist_mm(eef[0], eef[-1]), 2),
        "eef_path_len_mm": round(sum(step_moves), 2),
        "lid_total_move_success_to_end_mm": round(dist_mm(lid[0], lid[-1]), 2),
        "lid_path_len_mm": round(sum(lid_moves), 2),
        "success_predicate_frac_held_post": round(frac_hold, 3),
        "lid_z_rise_post_mm": round(1000.0 * (lid[-1][2] - lid[0][2]), 2),
        "lid_xy_to_closed_start_mm": round(1000.0 * post[0]["predicates"]["lid_xy_to_closed_pos"], 2),
        "lid_xy_to_closed_end_mm": round(1000.0 * post[-1]["predicates"]["lid_xy_to_closed_pos"], 2),
    })
    # drift-to-next-skill signal
    if skill == "grasp":
        out["drift_next"] = f"lid rose {out['lid_z_rise_post_mm']}mm after grasp (toward MOVE)"
    elif skill == "move_holding":
        out["drift_next"] = ("lid xy->closed %.0f->%.0fmm after preplace (toward PLACE)"
                             % (out["lid_xy_to_closed_start_mm"], out["lid_xy_to_closed_end_mm"]))
    else:
        out["drift_next"] = f"eef moved {out['eef_total_move_success_to_end_mm']}mm after place-success (re-grasp?)"
    boundary = out["eef_total_move_success_to_end_mm"] < 20.0 and frac_hold >= 0.90
    out["verdict"] = "BOUNDARY_RESPECTED" if boundary else "KEEPS_MOVING"
    return out


def main():
    results = {}
    for d in sys.argv[1:]:
        d = Path(d)
        for jf in sorted(d.glob("*_steps.jsonl")):
            skill = jf.name.replace("_steps.jsonl", "")
            if skill not in SUCCESS_PRED:
                continue
            steps = load_steps(jf)
            r = analyze_one(steps, skill)
            key = f"{d.name}:{skill}"
            results[key] = r
            print("%-32s ss=%-5s post=%-4s eef_tot=%-7s eef/step(mean/max)=%-14s pred_held=%-5s %s" % (
                key, r.get("success_step"), r.get("post_window_steps"),
                r.get("eef_total_move_success_to_end_mm"),
                str(r.get("eef_move_per_step_mm")), r.get("success_predicate_frac_held_post"),
                r.get("verdict")))
    outp = Path(sys.argv[1]).parent / "hold_analysis.json" if len(sys.argv) > 1 else Path("hold_analysis.json")
    json.dump(results, open(outp, "w"), indent=2)
    print("wrote", outp)


if __name__ == "__main__":
    main()
