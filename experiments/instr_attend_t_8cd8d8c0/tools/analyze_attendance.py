#!/usr/bin/env python3
"""Offline permutation-test analysis of the instruction-attendance probe (t_8cd8d8c0).

Reads probe_{state}_raw_full.json ([N, chunk, 12] per instruction) and computes, per fixed
obs state and per instruction (vs correct_full):
  within_L2        : mean pairwise L2 across the N samples (sampling noise floor for that instr)
  d_means          : L2 between this instruction's mean action chunk and correct_full's
  effect_size      : d_means / (global_noise_floor / sqrt(N))  -- ~1 = obs-only, >>1 = attends
  cosine           : cosine of the two mean chunks
  p                : permutation-test p (H0: label has no effect on the action)
Writes attendance_stats.json next to the inputs. Pure stdlib (no numpy on host).
"""
import json, math, random, itertools, sys
from pathlib import Path

CAT = {"correct_full": "correct", "skill_grasp": "skill", "skill_move": "skill", "skill_place": "skill",
       "opposite_open_drawer": "opposite", "irrelevant_pick_cup": "irrelevant",
       "irrelevant_turn_on_stove": "irrelevant", "empty": "degenerate", "nonsense": "degenerate"}


def flat(chunk):
    return [v for step in chunk for v in step]


def mean_vec(rows):
    n = len(rows); d = len(rows[0])
    return [sum(r[i] for r in rows) / n for i in range(d)]


def l2(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def within_mean(rows):
    ds = [l2(rows[i], rows[j]) for i, j in itertools.combinations(range(len(rows)), 2)]
    return sum(ds) / len(ds)


def cos(a, b):
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na > 1e-9 and nb > 1e-9 else float("nan")


def perm(A, B, iters=20000, seed=0):
    rng = random.Random(seed)
    obs = l2(mean_vec(A), mean_vec(B)); pool = A + B; n = len(A); c = 0
    for _ in range(iters):
        rng.shuffle(pool)
        if l2(mean_vec(pool[:n]), mean_vec(pool[n:])) >= obs:
            c += 1
    return obs, (c + 1) / (iters + 1)


def main():
    base = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    states = sys.argv[2].split(",") if len(sys.argv) > 2 else ["reset", "move", "place"]
    out = {}
    for st in states:
        p = base / f"probe_{st}_raw_full.json"
        if not p.exists():
            print(f"skip {st}: {p} missing"); continue
        full = json.loads(p.read_text())
        data = {l: [flat(c) for c in full[l]] for l in full}
        ref = data["correct_full"]; N = len(ref)
        nf = sum(within_mean(v) for v in data.values()) / len(data)
        scale = nf / math.sqrt(N)
        rows = []
        for l in data:
            if l == "correct_full":
                rows.append({"label": l, "category": CAT.get(l, "?"), "within_L2": round(within_mean(data[l]), 4),
                             "d_means": 0.0, "effect_size_vs_null": 0.0, "cosine": 1.0, "p": None}); continue
            obs, pv = perm(ref, [list(r) for r in data[l]])
            rows.append({"label": l, "category": CAT.get(l, "?"), "within_L2": round(within_mean(data[l]), 4),
                         "d_means": round(obs, 4), "effect_size_vs_null": round(obs / scale, 3),
                         "cosine": round(cos(mean_vec(ref), mean_vec(data[l])), 4), "p": pv})
        out[st] = {"N": N, "noise_floor_pairwise_L2": round(nf, 4),
                   "null_meandiff_scale": round(scale, 4), "rows": rows}
        sk = [r["effect_size_vs_null"] for r in rows if r["category"] == "skill"]
        off = [r["effect_size_vs_null"] for r in rows if r["category"] in ("opposite", "irrelevant")]
        print(f"{st:6s} N={N} noise_floor={nf:.3f}  skill_es={sum(sk)/len(sk):.2f}  offtask_es={sum(off)/len(off):.2f}")
    (base / "attendance_stats.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {base/'attendance_stats.json'}")


if __name__ == "__main__":
    main()
