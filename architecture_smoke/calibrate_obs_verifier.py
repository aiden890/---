"""Offline calibration + role-split sweep over cached VLM frame scores (t_32a4f9b6).

Consumes the ``cache.json`` produced by ``cache_frame_scores.py`` (one P(yes)
per corpus frame per candidate question, plus the offline sim GT label) and
answers, WITHOUT any GPU, every calibration question the task poses:

  * ROLE SPLIT -- boundary judgment vs success judgment are evaluated as two
    different operating points. The *boundary* judge may fire loosely (advance
    the skill); the *success* judge (does the whole task actually succeed) is
    the strict gate we tune to high precision.

  * PLACE FALSE-POSITIVE FIX, three levers, all replayed from the cache:
      1. per-skill ``hold_steps`` -- require the latched "yes" to persist for a
         run of consecutive scored frames (temporal stability: lid stays seated
         + gripper stays away AFTER release), instead of hold_steps=1.
      2. VQA decomposition -- combine the decomposed PLACE sub-questions
         (seated AND released/clear) with a min()/product rule instead of the
         single combined question.
      3. tau / latch (hysteresis_k) per skill.

  * METRICS -- per skill, at each operating point: confusion (TP/FP/FN/TN),
    precision / recall / F1, and the ROC/PR curve over tau. The whole point is
    to pick the (question-rule, tau, latch, hold_steps) that reaches a target
    PLACE precision (default 0.9) at maximum recall.

A rollout is scored by replaying its cached frames in order through the same
latch logic ``ObsVLMVerifier`` uses at runtime (K consecutive confident yes ->
ADVANCE), but here the per-frame P(yes) is looked up instead of computed, and an
extra ``hold_steps`` gate requires the yes-run to persist. The verifier's
ADVANCE step is then compared to the sim GT success step -- GT used as an
offline label only (obs-only invariant preserved: the label never feeds the
score rule).
"""
from __future__ import annotations

import argparse
import json
import itertools
from pathlib import Path


# --------------------------------------------------------------------------- #
#  score rules -- how per-frame P(yes) is derived from the cached questions     #
# --------------------------------------------------------------------------- #
def rule_single(qkey, view="full"):
    key = f"{qkey}@{view}"

    def f(scores):
        return scores.get(key)
    f.__name__ = f"single[{key}]"
    return f


def rule_and(qkeys, view="full", combine="min"):
    """AND of several sub-questions -> min (strict) or product of their P(yes)."""
    keys = [f"{q}@{view}" for q in qkeys]

    def f(scores):
        vals = [scores.get(k) for k in keys]
        if any(v is None for v in vals):
            return None
        if combine == "min":
            return min(vals)
        p = 1.0
        for v in vals:
            p *= v
        return p
    f.__name__ = f"and[{'+'.join(qkeys)}|{combine}|{view}]"
    return f


# --------------------------------------------------------------------------- #
#  replay one rollout through the latch + hold gate                             #
# --------------------------------------------------------------------------- #
def replay_rollout(frames, score_fn, *, tau, hysteresis_k, hold_steps):
    """Return the env_step at which the verifier would ADVANCE, or None.

    latch: hysteresis_k consecutive scored frames with P(yes) >= tau arms the
    'recognised' state. hold_steps: the yes-run must then persist for at least
    hold_steps consecutive scored frames total (>= max(hysteresis_k, hold_steps)
    consecutive yes) before ADVANCE fires -- the temporal-stability requirement
    that a single-frame flush at the release instant cannot satisfy.
    Any frame with P(yes) < tau resets the run (a confident 'no' breaks a
    transient false positive).
    """
    need = max(int(hysteresis_k), int(hold_steps))
    run = 0
    for fr in frames:
        p = score_fn(fr["scores"])
        if p is None:
            continue
        if p >= tau:
            run += 1
            if run >= need:
                return fr["env_step"]
        else:
            run = 0
    return None


def roc_auc_binary(labels, scores):
    """Tie-aware ROC AUC: probability a random positive outranks a negative."""
    pos = [s for y, s in zip(labels, scores) if y]
    neg = [s for y, s in zip(labels, scores) if not y]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def parse_views(text):
    views = [v.strip() for v in text.split(",") if v.strip()]
    allowed = {"full", "left", "right", "eye"}
    bad = [v for v in views if v not in allowed]
    if not views or bad:
        raise ValueError(f"invalid views {bad or views}; allowed={sorted(allowed)}")
    return views


def rule_frame_auc(rollouts, score_fn):
    """Frame ROC AUC with sim GT used only as an offline temporal label.

    Frames before ``gt_success_step`` are negatives even in an eventually
    successful episode. This measures transition timing rather than merely
    separating episodes that happened to end successfully.
    """
    labels, scores = [], []
    for ro in rollouts:
        success_step = ro.get("gt_success_step")
        for fr in ro["frames"]:
            score = score_fn(fr["scores"])
            if score is None:
                continue
            labels.append(success_step is not None and fr["env_step"] >= success_step)
            scores.append(score)
    return roc_auc_binary(labels, scores)


def confusion(records):
    """Event-detection confusion, counting an early ADVANCE as FP *and* FN.

    An early latch is unsafe and also consumes the one transition opportunity,
    so it is both a false alarm and a missed valid boundary. Consequently event
    counts can sum to more than the number of episodes; ``n`` remains episodes.
    """
    tp = fp = fn = tn = early_fp = 0
    offsets = []
    for r in records:
        gt, adv = r["gt_success"], r["advance_step"] is not None
        if gt and adv:
            offset = r["advance_step"] - r["gt_success_step"]
            if offset < 0:
                fp += 1
                fn += 1
                early_fp += 1
            else:
                tp += 1
                offsets.append(offset)
        elif not gt and adv:
            fp += 1
        elif gt and not adv:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    return {
        "n": len(records), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "early_fp": early_fp,
        "precision": prec, "recall": rec, "f1": f1,
        "mean_timing_offset": (sum(offsets) / len(offsets)) if offsets else None,
        "timing_offsets": offsets,
    }


def eval_point(rollouts, score_fn, *, tau, hysteresis_k, hold_steps):
    recs = []
    for ro in rollouts:
        adv = replay_rollout(ro["frames"], score_fn, tau=tau,
                             hysteresis_k=hysteresis_k, hold_steps=hold_steps)
        recs.append({
            "rollout": ro["rollout"], "gt_success": ro["gt_success"],
            "gt_success_step": ro["gt_success_step"], "advance_step": adv,
        })
    return confusion(recs), recs


# --------------------------------------------------------------------------- #
#  candidate score-rules per skill                                              #
# --------------------------------------------------------------------------- #
def candidate_rules(skill, views):
    """Build every question-rule x requested-view candidate.

    The old implementation accepted ``views`` but silently hard-coded ``full``;
    that made a per-view cache impossible to sweep. Keep rule construction in
    this one calibration engine and make camera routing explicit in rule names.
    """
    rules = {}
    for view in views:
        if skill == "grasp":
            rules[f"grasp@{view}"] = rule_single("grasp", view=view)
        elif skill == "move_holding":
            rules[f"move@{view}"] = rule_single("move", view=view)
        elif skill == "place":
            rules.update({
                f"place_combined@{view}": rule_single("place_combined", view=view),
                f"place_seated_and_clear_min@{view}": rule_and(
                    ["place_seated", "place_clear"], view=view, combine="min"),
                f"place_seated_and_released_min@{view}": rule_and(
                    ["place_seated", "place_released"], view=view, combine="min"),
                f"place_seated_released_clear_min@{view}": rule_and(
                    ["place_seated", "place_released", "place_clear"],
                    view=view, combine="min"),
                f"place_seated_and_clear_prod@{view}": rule_and(
                    ["place_seated", "place_clear"], view=view, combine="product"),
            })
        else:
            raise ValueError(f"unsupported calibration skill: {skill}")
    return rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", default="full")
    ap.add_argument("--tau-grid", default="0.5,0.6,0.7,0.8,0.9,0.95,0.98")
    ap.add_argument("--latch-grid", default="2,3")
    ap.add_argument("--hold-grid", default="1,3,5,8,12,16,20")
    ap.add_argument("--place-precision-target", type=float, default=0.9)
    args = ap.parse_args()

    cache = json.loads(Path(args.cache).read_text())
    by_skill = {}
    for ro in cache["rollouts"]:
        by_skill.setdefault(ro["skill"], []).append(ro)

    taus = [float(x) for x in args.tau_grid.split(",")]
    latches = [int(x) for x in args.latch_grid.split(",")]
    holds = [int(x) for x in args.hold_grid.split(",")]

    result = {"meta": cache["meta"], "skills": {}}
    views = parse_views(args.views)
    for skill, rollouts in by_skill.items():
        rules = candidate_rules(skill, views)
        rule_auc = {name: rule_frame_auc(rollouts, fn)
                    for name, fn in rules.items()}
        sweep = []
        for rname, rfn in rules.items():
            for tau, k, hold in itertools.product(taus, latches, holds):
                conf, _ = eval_point(rollouts, rfn, tau=tau,
                                     hysteresis_k=k, hold_steps=hold)
                sweep.append({
                    "rule": rname, "tau": tau, "hysteresis_k": k,
                    "hold_steps": hold, **{q: conf[q] for q in
                        ("n", "tp", "fp", "fn", "tn", "early_fp",
                         "precision", "recall", "f1", "mean_timing_offset")},
                })
        # pick best operating point: max recall subject to precision >= target
        target = args.place_precision_target if skill == "place" else 0.9
        feasible = [s for s in sweep if s["precision"] is not None
                    and s["precision"] >= target and s["recall"] is not None]
        # prefer highest recall, then highest precision, then smallest hold
        best = None
        if feasible:
            best = max(feasible, key=lambda s: (s["recall"], s["precision"],
                                                -s["hold_steps"]))
        else:
            # no point hits target: report best precision achievable
            cand = [s for s in sweep if s["precision"] is not None]
            if cand:
                best = max(cand, key=lambda s: (s["precision"], s["recall"] or 0))
        result["skills"][skill] = {
            "n_rollouts": len(rollouts),
            "n_pos": sum(1 for r in rollouts if r["gt_success"]),
            "n_neg": sum(1 for r in rollouts if not r["gt_success"]),
            "target_precision": target,
            "best": best,
            "roc_auc_by_rule": rule_auc,
            "sweep": sweep,
        }
        b = best or {}
        print(f"[{skill}] pos={result['skills'][skill]['n_pos']} "
              f"neg={result['skills'][skill]['n_neg']} best_rule={b.get('rule')} "
              f"tau={b.get('tau')} k={b.get('hysteresis_k')} hold={b.get('hold_steps')} "
              f"prec={b.get('precision')} rec={b.get('recall')} "
              f"fp={b.get('fp')} offset={b.get('mean_timing_offset')}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
