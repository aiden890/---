#!/usr/bin/env python3
"""Regenerate tracking/rlenv.json (RL-env reward-synced viewer) from the CURRENT
reward policy (rl-env-t_4f3f2b20/src/reward.py, use_milestones=False + HoldConfig).

Single source of truth: this script IMPORTS the real RewardManager / hold_step_reward /
SkillMonitor and REPLAYS existing base-policy rollout step-logs (full simulator
predicates incl. eef_pos) through them. It re-derives nothing about geometry or reward
and duplicates no logic -- it just plays recorded predicate streams back through the
production reward code, so the viewer shows exactly what training would pay.

Per-skill reward reproduced exactly as grpo_train_loop._run_one_skill:
  reward_sum += rb.primary            # RewardManager.step_reward (terminal-once + penalties)
  reward_sum += hold_step_reward(...) # post-success boundary shaping (training-only)
  if outcome is SUCCESS: reward_sum += 1.0   # per-skill terminal unit bonus (train loop)

Outputs the viewer schema (predicate_labels + clips[{label,case,goal,mp4,outcome,
success,total,n,success_step,targets,timeline[]}]) with each timeline step carrying the
reward BREAKDOWN under `d` = {term, hold, pen} plus cumulative `t` and fired notes.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "rl-env-t_4f3f2b20" / "src"))

from reward import RewardConfig, RewardManager, HoldConfig, hold_step_reward, official_success  # noqa: E402
from skill_manager import SkillMonitor, MonitorConfig, Skill, SkillOutcome  # noqa: E402

SKILL = {"grasp": Skill.GRASP, "move_holding": Skill.MOVE_HOLDING, "place": Skill.PLACE}
HORIZON = {"grasp": 300, "move_holding": 300, "place": 400}

# Predicates the viewer table shows (labels live in predicate_labels below).
SHOW_PREDS = [
    "lid_grasped", "lid_lifted", "lid_on_blender", "lid_upright_7deg",
    "gripper_lid_far_0.15", "gripper_lid_contact", "in_preplace_region", "eef_lid_dist",
]

PREDICATE_LABELS = {
    "lid_grasped": "\ub6da\uaecc \uc7a1\uc74c",
    "lid_lifted": "\ub4e4\uc5b4\uc62c\ub9bc",
    "lid_on_blender": "\ube14\ub80c\ub354 \uc704 \uc548\ucc29",
    "lid_upright_7deg": "\uc218\ud3c9(\u22647\u00b0)",
    "gripper_lid_far_0.15": "\uadf8\ub9ac\ud37c \uc774\uaca9(>15cm)",
    "gripper_lid_contact": "\uadf8\ub9ac\ud37c-\ub6da\uaecc \uc811\ucd09",
    "in_preplace_region": "\ub193\uae30 \uc9c1\uc804 \uc704\uce58",
    "eef_lid_dist": "\uc190-\ub6da\uaecc \uac70\ub9ac(m)",
}

# Per-skill reward-target checklist (mirrors the current reward: ONE terminal +1.0,
# no intermediate milestones, plus training-only hold shaping). `key` marks how the
# viewer lights it up: __term = paid when the skill-terminal +1.0 fired.
TARGETS = {
    "grasp": [
        {"name": "\ub6da\uaecc\uc744 \uc548\uc815\uc801\uc73c\ub85c \uc9d1\uc74c(lid_grasped 20\uc2a4\ud15d \uc5f0\uc18d)", "bonus": 1.0, "key": "__term"},
        {"name": "\uc131\uacf5 \ud6c4 \uadf8 \uc790\ub9ac \uc815\uc9c0(hold shaping)", "bonus": 0.05, "key": "__hold"},
    ],
    "move_holding": [
        {"name": "preplace \uc601\uc5ed \ub3c4\ub2ec(grasped AND in_preplace 3\uc2a4\ud15d)", "bonus": 1.0, "key": "__term"},
        {"name": "\uc131\uacf5 \ud6c4 \uc815\uc9c0(hold shaping)", "bonus": 0.05, "key": "__hold"},
    ],
    "place": [
        {"name": "official \uc131\uacf5: \uc548\ucc29 AND \uc774\uaca9>0.15m AND \uc218\ud3c9\u22647\u00b0", "bonus": 1.0, "key": "__term"},
        {"name": "\uc131\uacf5 \ud6c4 \uc815\uc9c0(hold shaping)", "bonus": 0.05, "key": "__hold"},
    ],
}


def load_steps(path: Path):
    rows = [json.loads(l) for l in path.open()]
    steps = [r for r in rows if r.get("type") == "step"]
    frames = [r for r in rows if r.get("type") == "frame"]
    return steps, frames


def frame_map(steps, frames):
    """step -> video frame index. Use recorded frame records (forward-filled) when
    present; otherwise fall back to the stride-2 convention f = step // 2 (matches the
    saved mp4s and the prior viewer)."""
    fm = {}
    if frames:
        last = 0
        recs = {r["step"]: r["frame_index"] for r in frames}
        for s in steps:
            if s["step"] in recs:
                last = recs[s["step"]]
            fm[s["step"]] = last
    else:
        for s in steps:
            fm[s["step"]] = s["step"] // 2
    return fm


def _success_now(skill, p):
    """The skill's own success predicate -- identical to grpo_train_loop._success_now."""
    if skill is Skill.GRASP:
        return bool(p.get("lid_grasped"))
    if skill is Skill.MOVE_HOLDING:
        return bool(p.get("lid_grasped")) and bool(p.get("in_preplace_region"))
    return bool(p.get("official_check_success")) or official_success(p)


def _find_outcome(skill, steps, horizon):
    """First-pass replay of the SkillMonitor FSM to find the natural termination:
    (outcome, outcome_step, success_step). Mirrors _run_one_skill's break semantics --
    the skill unit ENDS at its first terminal outcome (SUCCESS or a failure verdict);
    TIMEOUT if the horizon is reached with no verdict."""
    mon = SkillMonitor(skill, MonitorConfig())
    outcome, outcome_step, success_step = None, None, None
    for s in steps:
        o = mon.update(s["predicates"], s["step"], horizon)
        if o is SkillOutcome.SUCCESS and success_step is None:
            success_step = s["step"]
        if o:
            outcome, outcome_step = o, s["step"]
            break
    if outcome is None:
        outcome = SkillOutcome.TIMEOUT
        outcome_step = steps[-1]["step"]
    return outcome, outcome_step, success_step


def replay(skill_key, steps, frames, *, hold_steps=0):
    """Replay one recorded skill rollout through the REAL reward + monitor code.

    Reward is accrued over the skill unit's ACTIVE window only -- exactly as
    grpo_train_loop._run_one_skill, which breaks at the first terminal outcome (and,
    when hold_steps>0, continues for a post-success hold window). Steps past the active
    window are still kept in the timeline (so the video scrubber and the post-success
    "drift" behaviour remain visible) but contribute no further reward: their `d` is
    zero and cumulative `t` is frozen. Returns (timeline, outcome, success_step, total).
    """
    skill = SKILL[skill_key]
    horizon = HORIZON[skill_key]
    outcome, outcome_step, success_step = _find_outcome(skill, steps, horizon)

    # active reward window end (inclusive):
    #   success + hold : success_step + hold_steps (clamped to log length)
    #   success only   : success_step
    #   failure/timeout: outcome_step
    if success_step is not None:
        active_end = success_step + hold_steps if hold_steps > 0 else success_step
    else:
        active_end = outcome_step
    active_end = min(active_end, steps[-1]["step"])

    # Training-only shaping constants -- identical to grpo_train_loop.train_iteration:
    #   approach_coef 0.1 for GRASP only; flat timeout cost 0.5 on TIMEOUT.
    approach_coef = 0.0  # approach shaping disabled (operator 2026-09-16): success + hold only
    timeout_penalty = 0.5

    cfg = RewardConfig(horizon=horizon, use_milestones=False)
    rm = RewardManager(cfg)
    hold_cfg = HoldConfig()
    fm = frame_map(steps, frames)

    cum = 0.0
    prev_eef = list(steps[0]["predicates"].get("eef_pos", (0.0, 0.0, 0.0)))
    prev_dist = float(steps[0]["predicates"].get("eef_lid_dist", 0.0)) if approach_coef > 0.0 else None
    timeline = []

    for s in steps:
        step = s["step"]
        p = s["predicates"]
        skill_term = 0.0
        term = 0.0
        pen = 0.0
        hold = 0.0
        shape = 0.0

        if step <= active_end:
            # env done/trunc are False for a skill unit mid-episode (the SkillMonitor, not
            # the env, ends the skill), so reward.py's internal end-penalty never fires here
            # -- matching _run_one_skill, which passes the env's real (False) done/trunc.
            rb = rm.step_reward(step, p, done=False, truncated=False)
            term = round(rb.terminal, 4)
            pen = round(rb.penalty, 4)
            cum += rb.primary
            # GRASP approach shaping: reward += 0.1 * (prev_dist - curr_dist)
            if approach_coef > 0.0 and prev_dist is not None:
                curr_dist = float(p.get("eef_lid_dist", prev_dist))
                shape = round(approach_coef * (prev_dist - curr_dist), 4)
                cum += shape
                prev_dist = curr_dist
            # per-skill terminal unit bonus (+1.0) on first SUCCESS -- train loop's r += 1.0
            if success_step == step:
                skill_term = 1.0
                cum += 1.0
            # post-success boundary hold shaping (training-only)
            if hold_steps > 0 and success_step is not None and step > success_step:
                curr_eef = list(p.get("eef_pos", prev_eef))
                hold = round(hold_step_reward(prev_eef, curr_eef, _success_now(skill, p), hold_cfg), 4)
                cum += hold
            # flat timeout cost, added once at the terminal step of a TIMEOUT skill
            if step == active_end and outcome is SkillOutcome.TIMEOUT:
                pen = round(pen - timeout_penalty, 4)
                cum += -timeout_penalty
        prev_eef = list(p.get("eef_pos", prev_eef))

        fire = []
        if skill_term:
            fire.append("\uc2a4\ud0ac \uc131\uacf5 +1.0")
        if term:
            fire.append(f"terminal(decay) +{term}")
        if shape:
            fire.append(f"\uc811\uadfc shaping {'+' if shape > 0 else ''}{shape}")
        if pen:
            fire.append(f"\uac10\uc810 {pen}")
        if hold > 0:
            fire.append(f"\uc815\uc9c0 \ud648\ub4dc +{hold}")
        elif hold < 0:
            fire.append(f"\ud45c\ub958 \ud648\ub4dc {hold}")

        d = {"term": round(skill_term + term, 4), "hold": hold, "pen": pen, "shape": shape}
        entry = {
            "s": step, "f": fm[step], "t": round(cum, 4),
            "d": d, "fire": fire,
            "phase": ("post_success" if (success_step is not None and step >= success_step) else "pre_success"),
            "p": {k: (round(p[k], 4) if isinstance(p.get(k), (int, float)) and not isinstance(p.get(k), bool) else p.get(k)) for k in SHOW_PREDS},
        }
        if step > active_end:
            entry["inactive"] = True   # skill already ended; frames are raw-rollout tail
        timeline.append(entry)

    # terminal note at the reward-active end step
    for e in timeline:
        if e["s"] == active_end:
            if success_step is not None:
                e["end"] = {"note": f"\uc2a4\ud0ac \uc131\uacf5(step {success_step})", "total": round(cum, 4)}
            else:
                e["end"] = {"note": f"\uc2e4\ud328 \uc885\ub8cc({outcome.name}, step {outcome_step})", "total": round(cum, 4)}
            break
    return timeline, outcome, success_step, round(cum, 4)


# ---- clip catalogue: (label, case, outcome-desc, mp4, backing step-log, hold?) ----
CLIPS = [
    dict(label="grasp", case="\uc131\uacf5: lid_grasped 20\uc2a4\ud15d \uc5f0\uc18d",
         goal="\ube14\ub80c\ub354 \ub6da\uaecc\uc744 \uc548\uc815\uc801\uc73c\ub85c \uc9d1\uae30",
         instr="Pick up the blender lid securely.",
         mp4="rlenv/grasp_call8.mp4",
         log="rl-env-t_4f3f2b20/results/baseline-run1/grasp_call8_steps.jsonl", hold=0),
    dict(label="grasp", case="\uc2e4\ud328(TIMEOUT): \uc811\uadfc\ud588\ub2e4 \ubabb \uc7a1\uace0 \ud6c4\ud1f4",
         goal="\ube14\ub80c\ub354 \ub6da\uaecc\uc744 \uc548\uc815\uc801\uc73c\ub85c \uc9d1\uae30",
         instr="Pick up the blender lid securely.",
         mp4="rlenv/grasp.mp4",
         log="rollouts-xiaomi-t_4a072806/run1/grasp_steps.jsonl", hold=0),
    dict(label="move_holding", case="\uc131\uacf5: preplace \uc601\uc5ed \ub3c4\ub2ec",
         goal="\uc7a1\uc740 \ub6da\uaecc\uc744 \ube14\ub80c\ub354 \uc704\ub85c \ucda9\ub3cc \uc5c6\uc774 \uc774\ub3d9",
         instr="Move the grasped blender lid above the blender without colliding with the surroundings.",
         mp4="rlenv/move_holding.mp4",
         log="rollouts-xiaomi-t_4a072806/run1/move_holding_steps.jsonl", hold=0),
    dict(label="place", case="\uc2e4\ud328(OFF_TARGET): \uc0d0\ub69c\uac8c \ub193\uc784",
         goal="\ub6da\uaecc\uc744 \ube14\ub80c\ub354\uc5d0 \uc548\ucc29\u00b7\uc774\uaca9\u00b7\uc218\ud3c9\ub85c \ub193\uae30",
         instr="Place the grasped blender lid securely on top of the blender, release it, and move the gripper away.",
         mp4="rlenv/place.mp4",
         log="rollouts-xiaomi-t_4a072806/run1/place_steps.jsonl", hold=0),
    dict(label="place", case="\uc2e4\ud328(TIMEOUT): \ubabb \ub193\uc74c",
         goal="\ub6da\uaecc\uc744 \ube14\ub80c\ub354\uc5d0 \uc548\ucc29\u00b7\uc774\uaca9\u00b7\uc218\ud3c9\ub85c \ub193\uae30",
         instr="Place the grasped blender lid securely on top of the blender, release it, and move the gripper away.",
         mp4="rlenv/place_call5.mp4",
         log="rl-env-t_4f3f2b20/results/baseline-run1/place_call5_steps.jsonl", hold=0),
    # A PLACE SUCCESS clip (from the hold rollout, truncated at + shown WITH hold shaping)
    dict(label="place", case="\uc131\uacf5: official \uc131\uacf5(\uc548\ucc29+\uc774\uaca9+\uc218\ud3c9) \u00b7 \uc131\uacf5 \ud6c4 hold",
         goal="\ub6da\uaecc\uc744 \ube14\ub80c\ub354\uc5d0 \uc548\ucc29\u00b7\uc774\uaca9\u00b7\uc218\ud3c9\ub85c \ub193\uae30",
         instr="Place the grasped blender lid securely on top of the blender, release it, and move the gripper away.",
         mp4="rlenv/place_hold.mp4",
         log="tracking/rlenv/place_hold_steps.jsonl", hold=100),
]


def main():
    out = {"predicate_labels": PREDICATE_LABELS, "clips": []}
    print("=== regenerating rlenv.json from current reward policy ===")
    for spec in CLIPS:
        log = REPO / spec["log"]
        steps, frames = load_steps(log)
        timeline, outcome, succ_step, total = replay(
            spec["label"], steps, frames, hold_steps=spec["hold"])
        n = len(steps)
        clip = {
            "label": spec["label"],
            "case": spec["case"],
            "goal": spec["goal"],
            "instr": spec["instr"],
            "mp4": spec["mp4"],
            "overlay_mp4": str(Path(str(spec["mp4"])).with_name(
                Path(str(spec["mp4"])).stem + "_reward_overlay.mp4")),
            "fps": 20,
            "stride": 2,
            "outcome": outcome.name if hasattr(outcome, "name") else str(outcome),
            "success": succ_step is not None,
            "success_step": succ_step,
            "total": total,
            "n": n,
            "targets": [{"name": t["name"], "bonus": t["bonus"], "key": t["key"]} for t in TARGETS[spec["label"]]],
            "timeline": timeline,
        }
        out["clips"].append(clip)
        print(f"  {spec['label']:12} {clip['outcome']:10} n={n:3} success_step={succ_step} total={total:>8}  mp4={spec['mp4']}")

    dest = HERE / "rlenv.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {dest} ({dest.stat().st_size} bytes, {len(out['clips'])} clips)")


if __name__ == "__main__":
    main()
