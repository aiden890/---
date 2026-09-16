# GRASP GRPO — ONLINE adaptive-curriculum (moving band) sweep (t_2907de4f)

Redesign (supersedes upfront prefilter): per-iter group n_succ folds into a per-seed difficulty EMA; the sampler biases each iter's seed toward the MID band [1,7]/8 (mixed group = real GRPO signal). The band MOVES with the policy; eval/heldout FIXED.

Pre-filter baseline (operator-stopped) GATED ratio: **17/25 (68%)** (68% of iters threw away their GRPO signal).


## Arm `adaptive_sgd2e3` — SGD 2e-3 (v4)

- Iters: 30 | GATED (no step): **10/30 (33%)** | skipped: 0 | effective updates: **20**
- Group composition: `{'all_success': 5, 'all_failure': 6, 'mixed': 19}`  (**mixed=19/30 = 63%** = real GRPO advantage signal)
- Curriculum draw source: `{'explore': 14, 'exploit': 16}` (exploit = drew a known in-band seed)
- Band membership drift (seen/below/in/above): start `{'seen': 1, 'below': 0, 'in_band': 0, 'above': 1}` -> end `{'seen': 14, 'below': 6, 'in_band': 3, 'above': 5}` (in_band growing = curriculum found mid-difficulty seeds)
- Total n_success_hold across iters: **118** (>0 = the boundary-hold reward actually fired)
- Post-step (effective-update mean): ratio=1.0118 kl=0.32666 clip=0.9094 adapter_dL2=0.00100
- Curriculum touched **14** distinct seeds; final EMA-difficulty histogram (rounded n_succ): `{0: 6, 4: 1, 5: 1, 6: 1, 8: 5}`
- EVAL(before): official=0.05 grasp=0.6
- EVAL(after):  official=0.05 grasp=0.55
- HELDOUT:      official=0.1 grasp=0.55
- **grasp before/after delta: -0.0500 -> REGRESSED**

## Arm `adaptive_adamw1e5` — AdamW 1e-5 (amp_csi)

- Iters: 30 | GATED (no step): **11/30 (36%)** | skipped: 0 | effective updates: **19**
- Group composition: `{'all_success': 5, 'all_failure': 6, 'mixed': 19}`  (**mixed=19/30 = 63%** = real GRPO advantage signal)
- Curriculum draw source: `{'explore': 14, 'exploit': 16}` (exploit = drew a known in-band seed)
- Band membership drift (seen/below/in/above): start `{'seen': 1, 'below': 0, 'in_band': 0, 'above': 1}` -> end `{'seen': 14, 'below': 6, 'in_band': 3, 'above': 5}` (in_band growing = curriculum found mid-difficulty seeds)
- Total n_success_hold across iters: **116** (>0 = the boundary-hold reward actually fired)
- Post-step (effective-update mean): ratio=1.0015 kl=0.24292 clip=0.8889 adapter_dL2=0.00380
- Curriculum touched **14** distinct seeds; final EMA-difficulty histogram (rounded n_succ): `{0: 6, 4: 2, 6: 1, 8: 5}`
- EVAL(before): official=0.05 grasp=0.55
- EVAL(after):  official=0.15 grasp=0.6
- HELDOUT:      official=0.1 grasp=0.55
- **grasp before/after delta: +0.0500 -> IMPROVED**

---
Honest note: the curriculum's job is to REDUCE the GATED ratio (more mixed iters = more usable GRPO signal) vs the 68% baseline. A FLAT before==after with mixed% high means the sampler delivered signal but the update did not move the eta=0 ODE behaviour — a learning-strength/advantage deficit, NOT a no-signal verdict. Report improvement only if EVAL(after) > EVAL(before) on the FIXED eval set.
