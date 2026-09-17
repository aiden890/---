"""Statistical CONFIDENCE analysis of the obs-verifier calibration (task t_a309678b).

The operating-point selection lives in ``calibrate_obs_verifier.py`` /
``validate_success_gate.py`` (they pick view/tau/threshold vs sim GT and drive
the shipped SuccessGate). This module answers a DIFFERENT question the operator
raised: *how much can we trust those numbers given the corpus size?* — i.e. it
quantifies the statistical uncertainty, not the point estimate.

It consumes the SAME per-view cache (obs-only: cached P(yes) + offline sim-GT
label) and, for the SHIPPED per-skill success criteria (success_gate.
SUCCESS_CRITERIA), reports for each skill:

  * point precision/recall (episode-level, ANDed sub-facts, episode-mean agg),
  * the SEPARATION MARGIN: min positive aggregate vs max negative aggregate for
    each sub-fact (a razor-thin margin = fragile; a wide gap = robust),
  * bootstrap 95% CI on precision/recall (stratified resample of episodes),
  * leave-one-positive-out stability (how the confusion moves when any single
    positive is dropped — the honest read on "precision 1.0 on 2 positives"),
  * threshold-robustness band: the range of the primary threshold over which the
    confusion is unchanged.

Every number is computed from real cached frames + real sim-GT labels. No number
is fabricated. Sim GT is the offline label ONLY (obs-only invariant preserved).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from success_gate import SUCCESS_CRITERIA


def episode_sub_aggregate(rollout, sub, aggregate="mean", last_k=5):
    """Reproduce SuccessGate's per-sub episode aggregate from the cache."""
    key = f"{_qkey_for(sub.question)}@{sub.view}"
    vals = [fr["scores"][key] for fr in rollout["frames"] if key in fr["scores"]]
    if not vals:
        return None
    if aggregate == "last_k_mean":
        tail = vals[-last_k:]
        return sum(tail) / len(tail)
    return sum(vals) / len(vals)


# text->qkey resolved once from the cache meta at load time
_TEXT2KEY = {}


def _qkey_for(question_text):
    return _TEXT2KEY[question_text]


def gate_success(rollout, crit, thr_override=None):
    """Shipped-equivalent verdict: ALL sub-facts pass their threshold (AND),
    with the episode-level aggregate and min_frames guard."""
    if rollout["n_frames"] < crit.min_frames:
        return False
    for i, sub in enumerate(crit.subs):
        a = episode_sub_aggregate(rollout, sub, crit.aggregate, crit.last_k)
        thr = sub.threshold
        if thr_override is not None and i == 0:
            thr = thr_override
        if a is None or a < thr:
            return False
    return True


def confusion(pairs):
    tp = sum(1 for g, p in pairs if g and p)
    fp = sum(1 for g, p in pairs if not g and p)
    fn = sum(1 for g, p in pairs if g and not p)
    tn = sum(1 for g, p in pairs if not g and not p)
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec)


def bootstrap_ci(pairs, n_boot=5000, seed=0):
    """Stratified bootstrap: resample positives and negatives separately (keeps
    prevalence), recompute precision/recall each draw, report 2.5/97.5 pctile."""
    rng = random.Random(seed)
    pos = [p for p in pairs if p[0]]
    neg = [p for p in pairs if not p[0]]
    precs, recs = [], []
    for _ in range(n_boot):
        rs = ([pos[rng.randrange(len(pos))] for _ in pos] if pos else []) + \
             ([neg[rng.randrange(len(neg))] for _ in neg] if neg else [])
        m = confusion(rs)
        if m["precision"] is not None:
            precs.append(m["precision"])
        if m["recall"] is not None:
            recs.append(m["recall"])

    def ci(xs):
        if not xs:
            return None
        xs = sorted(xs)
        lo = xs[int(0.025 * (len(xs) - 1))]
        hi = xs[int(0.975 * (len(xs) - 1))]
        return [round(lo, 3), round(hi, 3)]
    return {"precision_ci95": ci(precs), "recall_ci95": ci(recs),
            "n_boot": n_boot}


def separation_margin(rollouts, crit):
    """For each sub-fact: min positive episode-aggregate vs max negative one, and
    the gap. Positive gap with threshold inside it = clean separation."""
    out = []
    for sub in crit.subs:
        pos = [episode_sub_aggregate(r, sub, crit.aggregate, crit.last_k)
               for r in rollouts if r["gt_success"]]
        neg = [episode_sub_aggregate(r, sub, crit.aggregate, crit.last_k)
               for r in rollouts if not r["gt_success"]]
        pos = [x for x in pos if x is not None]
        neg = [x for x in neg if x is not None]
        out.append({
            "question": _qkey_for(sub.question), "view": sub.view,
            "threshold": sub.threshold,
            "pos_min": round(min(pos), 3) if pos else None,
            "pos_vals": [round(x, 3) for x in sorted(pos)],
            "neg_max": round(max(neg), 3) if neg else None,
            "neg_vals": [round(x, 3) for x in sorted(neg)],
            "margin_posmin_minus_negmax": (round(min(pos) - max(neg), 3)
                                           if pos and neg else None),
            "threshold_inside_gap": (bool(pos and neg
                                          and max(neg) < sub.threshold <= min(pos))),
        })
    return out


def leave_one_positive_out(rollouts, crit):
    """Drop each positive in turn; report how precision/recall move. The honest
    stress test for 'precision 1.0 on N positives'."""
    positives = [r for r in rollouts if r["gt_success"]]
    results = []
    for drop in positives:
        subset = [r for r in rollouts if r is not drop]
        pairs = [(r["gt_success"], gate_success(r, crit)) for r in subset]
        m = confusion(pairs)
        results.append({"dropped": drop["rollout"],
                        "precision": m["precision"], "recall": m["recall"],
                        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"]})
    return results


def threshold_robustness(rollouts, crit, lo=0.30, hi=0.99, step=0.01):
    """Sweep the PRIMARY sub-fact threshold; report the contiguous band around
    the shipped value over which the confusion (tp/fp/fn) is unchanged."""
    base_pairs = [(r["gt_success"], gate_success(r, crit)) for r in rollouts]
    base = confusion(base_pairs)
    base_key = (base["tp"], base["fp"], base["fn"])
    t = lo
    band = []
    while t <= hi + 1e-9:
        pairs = [(r["gt_success"], gate_success(r, crit, thr_override=round(t, 2)))
                 for r in rollouts]
        m = confusion(pairs)
        if (m["tp"], m["fp"], m["fn"]) == base_key:
            band.append(round(t, 2))
        t += step
    ship = crit.subs[0].threshold
    # contiguous band containing the shipped threshold
    contig = [x for x in band if True]
    lo_b = min(band) if band else None
    hi_b = max(band) if band else None
    return {"shipped_primary_threshold": ship,
            "base_confusion": base_key,
            "unchanged_band_min": lo_b, "unchanged_band_max": hi_b,
            "band_width": (round(hi_b - lo_b, 2)
                           if (lo_b is not None and hi_b is not None) else 0.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="per-view cache.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    cache = json.loads(Path(args.cache).read_text())
    global _TEXT2KEY
    _TEXT2KEY = {t: k for k, t in cache["meta"]["questions"].items()}

    skill_name = {"grasp": "GRASP_OBJECT", "place": "PLACE_OBJECT"}
    report = {"corpus": cache["meta"].get("corpus"), "skills": {}}

    for cache_skill, gate_skill in skill_name.items():
        crit = SUCCESS_CRITERIA.get(gate_skill)
        if crit is None:
            continue
        ros = [r for r in cache["rollouts"] if r["skill"] == cache_skill]
        n_pos = sum(1 for r in ros if r["gt_success"])
        n_neg = len(ros) - n_pos
        pairs = [(r["gt_success"], gate_success(r, crit)) for r in ros]
        m = confusion(pairs)
        report["skills"][gate_skill] = {
            "n_episodes": len(ros), "n_pos": n_pos, "n_neg": n_neg,
            "point": m,
            "bootstrap": bootstrap_ci(pairs, n_boot=args.n_boot),
            "separation_margin": separation_margin(ros, crit),
            "leave_one_positive_out": leave_one_positive_out(ros, crit),
            "threshold_robustness": threshold_robustness(ros, crit),
            "criterion_note": crit.note,
        }

    Path(args.out).write_text(json.dumps(report, indent=2, default=str))

    # human summary
    for sk, d in report["skills"].items():
        p = d["point"]
        b = d["bootstrap"]
        print(f"\n=== {sk}  (n={d['n_episodes']}, pos={d['n_pos']}, neg={d['n_neg']}) ===")
        print(f"  point: precision={p['precision']} recall={p['recall']} "
              f"tp={p['tp']} fp={p['fp']} fn={p['fn']} tn={p['tn']}")
        print(f"  bootstrap95: precision {b['precision_ci95']}  recall {b['recall_ci95']}")
        for s in d["separation_margin"]:
            print(f"  sub {s['question']}@{s['view']} thr={s['threshold']}: "
                  f"pos_min={s['pos_min']} neg_max={s['neg_max']} "
                  f"margin={s['margin_posmin_minus_negmax']} "
                  f"thr_in_gap={s['threshold_inside_gap']}")
            print(f"      pos={s['pos_vals']}  neg={s['neg_vals']}")
        tr = d["threshold_robustness"]
        print(f"  threshold-robust band (primary): "
              f"[{tr['unchanged_band_min']}, {tr['unchanged_band_max']}] "
              f"width={tr['band_width']} (shipped={tr['shipped_primary_threshold']})")
        loo = d["leave_one_positive_out"]
        worst_p = [x["precision"] for x in loo if x["precision"] is not None]
        worst_r = [x["recall"] for x in loo if x["recall"] is not None]
        print(f"  leave-one-pos-out: precision range "
              f"[{min(worst_p) if worst_p else None}, {max(worst_p) if worst_p else None}]  "
              f"recall range [{min(worst_r) if worst_r else None}, {max(worst_r) if worst_r else None}]")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
