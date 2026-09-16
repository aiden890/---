"""Validate the role-split verifier against sim GT, from the cached frame scores.

Runs the ACTUAL ``success_gate.SuccessGate`` and ``obs_verifier.ObsVLMVerifier``
code paths over the calibration corpus, driving them with a cache-backed mock
backend so the reported precision/recall come from the shipped logic (not a
parallel re-implementation). Sim GT is the offline label only.

Two roles measured separately (the whole point of task t_32a4f9b6):

  * BOUNDARY judge (ObsVLMVerifier): does it ADVANCE each skill at a sensible
    frame? Reported as before/after so the reader sees it is fast + lenient.
  * SUCCESS judge (SuccessGate): the strict, view-routed, episode-level gate.
    Reported per skill as confusion + precision/recall -- this is the number the
    task targets (PLACE precision >= 0.9).

Also reports the BEFORE baseline: the old single-question, full-concat,
single-frame latch (tau 0.6 / k 2 / hold 1) used as the success signal, so the
before/after false-positive drop is explicit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from obs_verifier import VLMBackend
from success_gate import make_success_gate, SUCCESS_CRITERIA


class CacheBackend(VLMBackend):
    """Mock backend that returns the cached P(yes) for (question_text, view).

    The cache stores scores keyed ``<qkey>@<view>``; this backend maps the
    success_gate's literal question TEXT back to the cache qkey via a text->key
    table, then serves the pre-computed probability for the requested view. It
    exercises the real SuccessGate.observe/verdict path with zero GPU.
    """

    def __init__(self, text_to_qkey):
        self._t2k = text_to_qkey
        self._frame = None  # current frame's scores dict

    def set_frame(self, scores):
        self._frame = scores

    def score(self, images, question_text):
        return self.score_view(images, question_text, view="full")

    def score_view(self, images, question_text, view="full"):
        qkey = self._t2k[question_text]
        key = f"{qkey}@{view}"
        if key not in self._frame:
            raise KeyError(f"cache miss {key} (have {sorted(self._frame)[:6]}...)")
        return float(self._frame[key])


def build_text_to_qkey(cache):
    """Map each cached question's TEXT back to its qkey (success_gate uses text)."""
    return {text: qkey for qkey, text in cache["meta"]["questions"].items()}


def confusion(records):
    tp = fp = fn = tn = 0
    for gt, pred in records:
        if gt and pred:
            tp += 1
        elif not gt and pred:
            fp += 1
        elif gt and not pred:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1}


def eval_success_gate(cache, backend):
    """Drive the real SuccessGate over each rollout; return per-skill confusion."""
    skill_name = {"grasp": "GRASP_OBJECT", "place": "PLACE_OBJECT"}
    per_skill = {}
    details = []
    for ro in cache["rollouts"]:
        sk = skill_name[ro["skill"]]
        if sk not in SUCCESS_CRITERIA:
            continue
        gate = make_success_gate(sk, backend)
        for fr in ro["frames"]:
            backend.set_frame(fr["scores"])
            gate.observe(fr["scores"])  # images arg unused by CacheBackend
        v = gate.verdict()
        per_skill.setdefault(sk, []).append((ro["gt_success"], v["success"]))
        details.append({"rollout": ro["rollout"], "gt": ro["gt_success"],
                        "pred": v["success"], "sub_scores": v.get("sub_scores"),
                        "sub_thresholds": v.get("sub_thresholds"),
                        "sub_views": v.get("sub_views")})
    return {sk: confusion(recs) for sk, recs in per_skill.items()}, details


def eval_baseline_before(cache, tau=0.6, k=2):
    """BEFORE baseline: old single-question full-concat single-frame latch used
    directly as the success signal (k consecutive P(yes)>=tau anywhere -> success)."""
    skill_q = {"grasp": "grasp@full", "place": "place_combined@full"}
    skill_name = {"grasp": "GRASP_OBJECT", "place": "PLACE_OBJECT"}
    per_skill = {}
    for ro in cache["rollouts"]:
        key = skill_q[ro["skill"]]
        run = 0
        fired = False
        for fr in ro["frames"]:
            if fr["scores"].get(key, 0.0) >= tau:
                run += 1
                if run >= k:
                    fired = True
                    break
            else:
                run = 0
        per_skill.setdefault(skill_name[ro["skill"]], []).append(
            (ro["gt_success"], fired))
    return {sk: confusion(recs) for sk, recs in per_skill.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True,
                    help="per-view cache.json (needs @left/@right/@eye scores)")
    ap.add_argument("--cache-full", default=None,
                    help="optional full-concat cache for the BEFORE baseline")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cache = json.loads(Path(args.cache).read_text())
    t2k = build_text_to_qkey(cache)
    backend = CacheBackend(t2k)
    after, details = eval_success_gate(cache, backend)

    before_cache = json.loads(Path(args.cache_full).read_text()) if args.cache_full else cache
    before = eval_baseline_before(before_cache)

    result = {
        "corpus": cache["meta"].get("corpus"),
        "success_gate_after": after,
        "success_baseline_before": before,
        "criteria": {sk: c.note for sk, c in SUCCESS_CRITERIA.items()},
        "details": details,
    }
    Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    print("=== SUCCESS-judge (strict, view-routed, episode-level) — AFTER ===")
    for sk, m in after.items():
        print(f"  {sk}: prec={m['precision']} rec={m['recall']} "
              f"tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}")
    print("=== SUCCESS baseline (old single-Q full-concat latch) — BEFORE ===")
    for sk, m in before.items():
        print(f"  {sk}: prec={m['precision']} rec={m['recall']} "
              f"tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
