#!/usr/bin/env python3
"""adaptive_aggregate.py — aggregate the two ONLINE adaptive-curriculum arms into a REPORT.

Reads results/adaptive/<run>/{run_summary.json,train_log.jsonl,eval_before.json,
eval_after.json,eval_heldout.json,seed_difficulty.json} for adaptive_sgd2e3 (v4) and
adaptive_adamw1e5 (amp_csi), and reports honestly:
  (a) GATED ratio (all-success/all-failure iters with no optimizer step) — the headline the
      adaptive curriculum is meant to reduce vs the pre-filter baseline (17/25 = 68% GATED),
  (b) group-composition curve over iters (does mixed% stay high as the policy shifts?),
  (c) EVAL before/after/heldout delta on the FIXED eval set = real policy improvement,
  (d) post-step ratio/KL/clip/adapter_dL2 update-strength diagnostics,
  (e) curriculum behaviour: explore/exploit split, band-membership drift (below/in/above
      over the run), and the final per-seed difficulty distribution.
No success-dressing: if before==after the report says "no improvement".

Usage: python3 adaptive_aggregate.py <results/adaptive dir>
"""
import json, sys, statistics
from pathlib import Path

base = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
ARMS = [("adaptive_sgd2e3", "SGD 2e-3", "v4"),
        ("adaptive_adamw1e5", "AdamW 1e-5", "amp_csi")]
# pre-filter baseline reference (operator-stopped run): 25 iters, 17 GATED.
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


print("# GRASP GRPO — ONLINE adaptive-curriculum (moving band) sweep (t_2907de4f)\n")
print("Redesign (supersedes upfront prefilter): per-iter group n_succ folds into a per-seed "
      "difficulty EMA; the sampler biases each iter's seed toward the MID band [1,7]/8 "
      "(mixed group = real GRPO signal). The band MOVES with the policy; eval/heldout FIXED.\n")
print(f"Pre-filter baseline (operator-stopped) GATED ratio: **{BASELINE_GATED}** "
      "(68% of iters threw away their GRPO signal).\n")

for run, label, server in ARMS:
    d = base / run
    print(f"\n## Arm `{run}` — {label} ({server})\n")
    tl = load_jsonl(d / "train_log.jsonl")
    diff = load_json(d / "seed_difficulty.json")
    before = load_json(d / "eval_before.json")
    after = load_json(d / "eval_after.json")
    heldout = load_json(d / "eval_heldout.json")

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
        mixed = comps.get("mixed", 0)
        print(f"- Group composition: `{comps}`  "
              f"(**mixed={mixed}/{n} = {100*mixed//max(n,1)}%** = real GRPO advantage signal)")
        # explore/exploit split from the curriculum source tag
        srcs = {}
        for m in tl:
            s = m.get("adaptive_source", "?")
            srcs[s] = srcs.get(s, 0) + 1
        print(f"- Curriculum draw source: `{srcs}` (exploit = drew a known in-band seed)")
        # band-membership drift: first vs last logged band_stats
        bs_first = next((m.get("adaptive_band_stats") for m in tl if m.get("adaptive_band_stats")), None)
        bs_last = next((m.get("adaptive_band_stats") for m in reversed(tl) if m.get("adaptive_band_stats")), None)
        if bs_first and bs_last:
            print(f"- Band membership drift (seen/below/in/above): "
                  f"start `{bs_first}` -> end `{bs_last}` "
                  f"(in_band growing = curriculum found mid-difficulty seeds)")
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

    if diff:
        dd = diff.get("difficulty", {})
        # final EMA distribution over all seeds the curriculum touched
        hist = {}
        for v in dd.values():
            k = round(v.get("ema", -1))
            hist[k] = hist.get(k, 0) + 1
        print(f"- Curriculum touched **{len(dd)}** distinct seeds; final EMA-difficulty "
              f"histogram (rounded n_succ): `{dict(sorted(hist.items()))}`")

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

print("\n---\nHonest note: the curriculum's job is to REDUCE the GATED ratio (more mixed "
      "iters = more usable GRPO signal) vs the 68% baseline. A FLAT before==after with "
      "mixed% high means the sampler delivered signal but the update did not move the eta=0 "
      "ODE behaviour — a learning-strength/advantage deficit, NOT a no-signal verdict. "
      "Report improvement only if EVAL(after) > EVAL(before) on the FIXED eval set.")
