#!/usr/bin/env python3
"""band_aggregate.py — aggregate the two band-sweep arms into a markdown REPORT.

Reads results/band/<run>/{run_summary.json,train_log.jsonl,eval_before.json,
eval_after.json,eval_heldout.json,train_pool.json} for band_sgd2e3 (v4) and
band_adamw1e5 (amp_csi), and reports honestly:
  (a) GATED ratio (all-success/all-failure iters with no optimizer step) — the headline
      the prefilter is meant to reduce vs the pre-filter baseline (17/25 = 68% GATED),
  (b) EVAL before/after/heldout delta on the FIXED eval set = real policy improvement,
  (c) post-step ratio/KL/clip/adapter_dL2 update-strength diagnostics,
  (d) the band-pool composition (nsucc histogram over scanned seeds).
No success-dressing: if before==after the report says "no improvement".

Usage: python3 band_aggregate.py <results/band dir>
"""
import json, sys, statistics
from pathlib import Path

base = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
ARMS = [("band_sgd2e3", "SGD 2e-3", "v4"), ("band_adamw1e5", "AdamW 1e-5", "amp_csi")]
# pre-filter baseline reference (from the operator's stopped run): 25 iters, 17 GATED.
BASELINE_GATED = "17/25 (68%)"


def load_jsonl(p):
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out


def load_json(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def fmt(x, d=4):
    if isinstance(x, (int, float)):
        return f"{x:.{d}f}"
    return str(x)


print("# GRASP GRPO — difficulty-band prefilter sweep (t_2907de4f)\n")
print("Redesign: train seeds prefiltered to a MID-difficulty base-success band "
      "(--train-difficulty-band 1 7 over group 8), drawn WITHOUT replacement; "
      "eval/heldout FIXED (not band-filtered).\n")
print(f"Pre-filter baseline (operator-stopped) GATED ratio: **{BASELINE_GATED}** "
      "(68% of iters threw away their GRPO signal).\n")

for run, label, server in ARMS:
    d = base / run
    print(f"\n## Arm `{run}` — {label} ({server})\n")
    tl = load_jsonl(d / "train_log.jsonl")
    pool = load_json(d / "train_pool.json")
    before = load_json(d / "eval_before.json")
    after = load_json(d / "eval_after.json")
    heldout = load_json(d / "eval_heldout.json")

    if pool:
        hist = pool.get("nsucc_hist", {})
        print(f"- Band pool: kept **{len(pool.get('train_seeds', []))}** seeds "
              f"(band {pool.get('band')}, scanned {pool.get('n_scanned')}/"
              f"{pool.get('scan_cap')}), base n_succ histogram over scanned seeds: "
              f"`{hist}`")

    if tl:
        n = len(tl)
        gated = [m for m in tl if m.get("gated")]
        skipped = [m for m in tl if m.get("skipped")]
        eff = [m for m in tl if not m.get("gated") and not m.get("skipped")]
        comps = {}
        for m in tl:
            c = m.get("group_composition", "?")
            comps[c] = comps.get(c, 0) + 1
        print(f"- Iters: {n} | GATED (no step): **{len(gated)}/{n} "
              f"({100*len(gated)//max(n,1)}%)** | skipped: {len(skipped)} | "
              f"effective updates: **{len(eff)}**")
        print(f"- Group composition: `{comps}`  "
              f"(mixed = real GRPO advantage signal)")
        nsucc_hold = sum(m.get("n_success_hold", 0) or 0 for m in tl)
        print(f"- Total n_success_hold across iters: **{nsucc_hold}** "
              f"(>0 = the boundary-hold reward actually fired)")
        if eff:
            def avg(key):
                vals = [m.get(key) for m in eff if isinstance(m.get(key), (int, float))]
                return statistics.mean(vals) if vals else None
            print(f"- Post-step (effective-update mean): "
                  f"ratio={fmt(avg('post_step_mean_ratio'))} "
                  f"kl={fmt(avg('post_step_mean_kl'),5)} "
                  f"clip={fmt(avg('post_step_clip_fraction'))} "
                  f"adapter_dL2={fmt(avg('adapter_delta_l2'),5)}")

    def rate(x):
        if not x:
            return "n/a"
        return (f"official={x.get('official_success_rate')} "
                f"grasp={x.get('grasp_success_rate')}")

    print(f"- EVAL(before): {rate(before)}")
    print(f"- EVAL(after):  {rate(after)}")
    print(f"- HELDOUT:      {rate(heldout)}")
    if before and after:
        b = before.get("grasp_success_rate")
        a = after.get("grasp_success_rate")
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            delta = a - b
            verdict = ("IMPROVED" if delta > 1e-9 else
                       "REGRESSED" if delta < -1e-9 else "FLAT (no improvement)")
            print(f"- **grasp before/after delta: {delta:+.4f} -> {verdict}**")

print("\n---\nHonest note: a FLAT before==after with n_success_hold>0 means GRPO SAW the "
      "signal (band prefilter worked) but the update did not move the eta=0 ODE behavior "
      "— a learning-strength/advantage deficit, NOT a no-signal verdict. Report as such.")
