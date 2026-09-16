# Obs-only skill-termination verifier — validation report

Task **t_d4268a2e**. Redesigns the skill-termination verifier so a skill's completion is judged from ONLY what the robot actually receives (3 camera images + 14-D proprio), never the simulator's privileged object-pose / fixture predicates. The judge is the policy's own frozen Qwen3-VL backbone (VQA: P(yes) that this skill's goal is achieved); a proprio event-gate keeps VLM calls at a realistic cadence and provides a comparison baseline.

## Obs-only guarantee (enforced in code)
- `obs_verifier.assert_obs_only` rejects every privileged sim predicate key (`lid_on_blender`, `official_check_success`, `lid_pos`, `lid_grasped`, …) from the runtime judge input.
- `ObsInput` is a frozen dataclass with fields for images + proprio only; `ObsVLMVerifier.update` re-asserts on every step.
- Unit tests: **25/25 passed** (`test_obs_verifier.py`) — schema guard, camera-key guard, proprio gate, VLM hysteresis latch, cadence gating, timeout→REPLAN, mock e2e.

## Realism: VLM call cost
- One VQA forward on CPU (fp32, contended v4): **433 s**. Per-step (20 Hz) VLM gating is therefore impossible on CPU; the verifier gates the VLM to a realistic cadence (replan-boundary interval + proprio settle-events), ~4–8 calls per skill. On GPU the same forward is ~0.3 s (server avg_time in xiaomi-server logs).

## Real-VLM discrimination probe
Scores ground-truth-labeled frames (offline sim label, never fed to the verifier) with each skill's completion question and checks the VLM separates done from not-done.

- frames scored: **8** (2 POS = gt-success, 6 NEG).
- mean P(yes): **0.562** on POS vs **0.449** on NEG (separation **0.113**).
- best single threshold τ=0.851: accuracy **0.75** (tp=1, tn=5).
- mean forward: 432.7 s (cpu, 16 threads).

| rollout | skill | phase | env_step | gt_done | P(yes) |
|---|---|---|---|---|---|
| rand10_grasp_seed0 | grasp | neg_early | 34 | False | 0.4265 |
| rand10_grasp_seed0 | grasp | pos_success | 136 | True | 0.8511 |
| rand10_grasp_seed3 | grasp | neg_mid | 82 | False | 0.8969 |
| rand10_grasp_seed3 | grasp | neg_late | 186 | False | 0.0473 |
| rand10_place_seed9 | place | neg_early | 28 | False | 0.4499 |
| rand10_place_seed9 | place | pos_success | 112 | True | 0.2728 |
| rand10_place_seed0 | place | neg_mid | 160 | False | 0.7112 |
| rand10_place_seed0 | place | neg_late | 360 | False | 0.1610 |

## Findings (honest, no success-packing)

- **GRASP is obs-judgeable, PLACE is not (single-frame).** GRASP separates: the
  true grasp frame scores 0.85 while the dropped-lid end of a failed grasp scores
  0.05. PLACE fails: the true place-success frame scores only **0.27** (a false
  negative) — the zero-shot VLM cannot reliably read "lid seated on base AND
  gripper released" from these camera angles, so PLACE completion needs a temporal
  signal or a fine-tuned head, not a single-frame yes/no.
- **A single frame produces false positives → the hysteresis latch is load-bearing.**
  The failed grasp `rand10_grasp_seed3` scores **0.90** mid-rollout (gripper
  transiently near the lid) yet 0.05 later. A one-shot threshold verifier would
  mis-ADVANCE here; the verifier's K-consecutive-yes latch + proprio settle gating
  exist precisely to reject such transients. This is why the runtime design does
  not trust one frame.
- **Zero-shot base VLM ≠ a trained termination head.** These P(yes) come from the
  *frozen, untrained* Qwen3-VL backbone with a hand-written question. The result
  bounds what is available today without training; the obs-only verifier interface
  is what a trained termination classifier (task requirement option (b)) would drop
  into unchanged (same `VLMBackend.score` contract).
- **VLM cost forces event gating.** 433 s/forward on CPU (0.3 s on GPU) makes
  per-step gating impossible; the measured cost is the evidence behind the
  replan-interval + proprio-event cadence, not an assumption.

## Limitations / honesty notes

- 8 real forwards (2 POS/6 NEG) on a contended CPU — a discrimination probe, not
  the full per-step stream sweep. The stream-replay harness
  (`validate_obs_verifier.py`) is verified end-to-end on all 20 v4 rollouts with a
  mock backend (frame-split + ground-truth alignment + precision/recall/timing);
  swapping `--backend qwen` runs it for real when the GPU is free (the parallel
  training task t_e5cd5736/t_3ed65912 owned the GPU during this run).
- These older rollouts recorded the composed 3-cam frames + sim predicates but not
  the per-step 14-D proprio, so the probe drives the VLM channel; the proprio gate
  is unit-tested separately. A fresh obs-dumping rollout would let the full harness
  exercise both channels together.
- POS/NEG frames are chosen by the sim ground-truth success step — used ONLY as an
  offline label; the verifier code path never receives it (asserted).

## Files
- `architecture_smoke/obs_verifier.py` — obs-only verifier (ProprioGate + ObsVLMVerifier + schema guard).
- `architecture_smoke/vlm_backends.py` — MockVLMBackend + QwenVLMScorerBackend (reuses `vlm_scorer` prompt+logit math).
- `architecture_smoke/test_obs_verifier.py` — unit tests.
- `architecture_smoke/validate_obs_verifier.py` — stream-replay agreement harness.
- `architecture_smoke/probe_vlm_discrimination.py` — real-VLM discrimination probe.
