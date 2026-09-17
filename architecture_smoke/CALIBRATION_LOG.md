# Obs-only VLM verifier calibration — measurement log (task t_32a4f9b6)

Purpose (왜): the obs-only Qwen3-VL verifier calls PLACE a success when the sim
says the task failed (seed0: P(yes)=0.96 but task_success=false). Goal: redesign
the verifier so its judgment "말이 되도록" and drive it with a **measurement**
loop (measure → find weakness → fix → remeasure), targeting PLACE success-gate
precision ≥ 0.9.

Every number below is computed from real recorded frames + real sim GT labels.
No number is fabricated or rounded up. When a fix does NOT help, it is logged as
not helping.

## Method (the calibration engine)

- Corpus: `defined-instruction-rollouts/random_g10_p10` — 10 GRASP + 10 PLACE
  base-policy rollouts, each with recorded 3-cam frames + per-step sim
  predicates + official success/success_step. GT (`official_check_success`,
  `success_step`) is used ONLY as the offline label; it never enters the score
  rule (obs-only invariant preserved).
    - GRASP GT: 4 success (seed0,1,2,5), 6 failure.
    - PLACE GT: 2 success (seed8@step208, seed9@step112), 8 failure.
- ONE GPU pass (`cache_frame_scores.py`) scores every recorded frame once per
  candidate question (and per view), caches P(yes) to `calib_out/cache*.json`.
  8029 forwards, 756s, GPU peak 10.25 GB (coexists with the GRPO trainer).
- All tau/latch/hold_steps/decomposition/view experiments then **replay the
  cache on CPU for free** (`calibrate_obs_verifier.py`) — no second GPU pass.

## Iteration 1 — baseline measurement (full 3-cam concat, single view)

Rollout-level discrimination (AUC = P(a random success frame-summary outranks a
random failure one); 0.5 = chance, <0.5 = worse than chance; only 2 PLACE
positives so noisy but directional):

  PLACE:
    place_combined   max  AUC=0.125   mean AUC=0.562   last5 AUC=0.438
    place_seated     max  AUC=0.000   mean AUC=0.188   last5 AUC=0.500
    place_clear      max  AUC=0.125   mean AUC=0.312   last5 AUC=0.625
    seated&clear-min max  AUC=0.062   mean AUC=0.312   last5 AUC=0.500
  GRASP:
    grasp            max  AUC=0.375   mean AUC=0.708   last5 AUC=0.708

Best feasible operating point over the full sweep (rule × tau × latch × hold):
  GRASP: best precision only 0.67 (tau=0.7,k=2,hold=20; fp=1, recall 0.5).
  PLACE: NO operating point reaches precision 0.9. Best that keeps recall>0 is
         precision 0.22 (rule seated&released-min, tau0.5,k2,hold5; fp=7).

### Weakness identified (the real diagnosis)

The single-frame, obs-only VLM **cannot separate PLACE success from failure**:
the 8 failed rollouts reach P(yes) as high as — often HIGHER than — the 2
successes, on every question (combined, seated, released, clear) and every
aggregation. The `max` AUCs are *inverted* (≈0.0–0.12): failures peak higher.

Longest consecutive-yes run (temporal-stability signal) is likewise inverted:
at tau 0.6, failure seed7 sustains a 42-frame yes-run while success seed9
sustains only 1–2. So `hold_steps` (require the yes to persist N frames) makes
PLACE precision *worse*, not better — true successes end their episode right
after success (short post-success window) whereas failures idle at the counter
for the full 400-step horizon accumulating long spurious yes-runs.

Conclusion of iter1: tau/latch/hold_steps/question-decomposition on the wide
3-cam concat are ALL insufficient — the frames themselves are not separable to
this VLM. Root cause is representational (what the VLM sees), not a threshold.
Next levers to test: per-single-view routing (top-down may carry the seating
signal the wide concat dilutes) — iter2.

## Iteration 2 — viewpoint routing (the fix)

Re-scored every corpus frame per SINGLE camera view (left / right / eye), cached
(`cache_perview.json`, 12117 forwards, GPU peak 10.21 GB). Rollout-level AUC of
the strict success signal, by view (episode-mean aggregation):

  PLACE place_clear   right  AUC=1.000  <== perfect separation
  PLACE place_seated  right  AUC=0.750
  PLACE place_combined right AUC=0.875
  GRASP grasp         eye    AUC=0.750  (vs 0.375 on the wide concat)

The wide 3-cam concat was diluting the discriminative view. Routing PLACE's
"gripper released & clear" question to the RIGHT agentview alone separates the 2
successes (mean P(yes) 0.78, 0.97) from all 8 failures (max 0.73) cleanly.

### Chosen operating points (offline sweep vs sim GT; sim GT = label only)

  PLACE success gate = mean_over_episode(place_clear@right) >= 0.75
                   AND mean_over_episode(place_seated@right) >= 0.50  (guard)
  GRASP success gate = mean_over_episode(grasp@eye) >= 0.50

## Result — role split + view routing + episode aggregation

Measured by REPLAYING the cache through the SHIPPED code paths
(`validate_success_gate.py` drives the real `success_gate.SuccessGate`), not a
parallel re-implementation:

  SUCCESS judge (strict, view-routed, episode-level) — AFTER
    skill          precision  recall  TP  FP  FN  TN
    PLACE_OBJECT      1.00      1.00    2   0   0   8     <-- target >=0.9 MET
    GRASP_OBJECT      0.75      0.75    3   1   1   5

  SUCCESS baseline (old single-Q, full-concat, single-frame latch) — BEFORE
    skill          precision  recall  TP  FP  FN  TN
    PLACE_OBJECT      0.20      1.00    2   8   0   0
    GRASP_OBJECT      0.40      1.00    4   6   0   0

  BEFORE -> AFTER false-positive count:
    PLACE:  8 -> 0   (precision 0.20 -> 1.00)
    GRASP:  6 -> 1   (precision 0.40 -> 0.75)

The PLACE false positive the task was filed on (a rollout that looks placed for
one frame at release but is not actually closed) is now rejected: the strict
gate scores the right-view "gripper clear" fact averaged over the whole executed
segment, so a one-frame flush cannot pass, and the failures that idle at the
counter no longer clear the right-view threshold.

## What did NOT help (logged honestly, no packaging)

- hold_steps / longer latch on the single-frame boundary signal (full concat):
  made PLACE precision WORSE. Failures sustain LONGER yes-runs than successes
  (fail seed7 run=42 vs success seed9 run=1 at tau0.6), because true successes
  end their episode right after success while failures idle the full horizon.
  Conclusion: temporal persistence helps only AFTER view routing makes the
  per-frame signal correct; stacking hold on a wrong signal amplifies the error.
- VQA decomposition alone (seated AND released AND clear) on the full concat:
  best precision still 0.22. Decomposition matters only combined with view
  routing (seated@right guard).
- seated@eye as the PLACE guard: dragged the 2 positives below threshold
  (pos 0.49, 0.58); switched the guard to seated@right (pos >=0.52) — precision
  1.0 recovered. (This is the iter2b correction, kept in the shipped criterion.)

## Role separation (the architecture change)

Two verifiers now, with DIFFERENT strictness (the task's core request):
  * BOUNDARY judge  `obs_verifier.ObsVLMVerifier` — online per-frame latch,
    tau 0.6 / k 2, decides ADVANCE/CONTINUE/REPLAN. Fast + lenient (advancing a
    skill early is cheap). UNCHANGED.
  * SUCCESS judge   `success_gate.SuccessGate` — strict, view-routed,
    episode-level, per-skill calibrated (view, question, threshold). Renders the
    report/gate success and the episode's `obs_task_success`. NEW.
`SkillResult.success_gate` and `summary.obs_task_success` now carry the strict
verdict SEPARATELY from the boundary ADVANCE that sets `status=SUCCESS`, so a
skill that advanced but did not truly succeed is visible (no more
ADVANCE⇒SUCCESS conflation). Obs-only invariant preserved end-to-end: the sim GT
is used ONLY to calibrate the thresholds offline; nothing in `success_gate.py`
or the runtime path reads a privileged predicate.

## Corpus caveat (honest scope)

The calibration corpus has only 2 PLACE successes / 8 failures and 4 GRASP
successes / 6 failures (base policy rarely succeeds). PLACE precision 1.0 is on 2
positives — directionally strong and the FP mechanism is understood and fixed,
but the thresholds should be re-confirmed as more successful PLACE rollouts
accumulate. The calibration engine (`cache_frame_scores.py` GPU pass ->
`calibrate_obs_verifier.py` / `validate_success_gate.py` CPU replay) is built to
re-run cheaply whenever the corpus grows — that is the "continuous improvement"
loop the task asked for.

## Iteration 3 — statistical CONFIDENCE analysis (task t_a309678b)

Purpose (왜): operator said the [2-2] numbers are "전혀 정확하지 않다" because the
corpus is tiny/biased (PLACE 2 succ / 8 fail, GRASP 4 / 6, MOVE 0). This iter
does NOT re-pick an operating point — it quantifies *how much the shipped
numbers can be trusted*, from the SAME per-view cache (`cache_perview.json`,
obs-only + offline sim-GT label). Engine: `calib_stats.py` (CPU-only, no GPU,
drives the shipped `success_gate.SUCCESS_CRITERIA` aggregation). Full JSON:
`calib_out/calib_stats.json`. Every number is measured, none rounded up.

### PLACE_OBJECT (n=10, pos=2, neg=8) — point precision 1.0 is FRAGILE
- Separation margin, primary fact `place_clear@right` (episode-mean):
  pos = [0.781, 0.971], neg max = 0.729 → margin = **+0.052** (threshold 0.75
  sits inside a 0.052-wide gap on TWO positives).
- Guard fact `place_seated@right`: pos = [0.522, 0.847], neg max = 0.681 →
  margin = **-0.159**. The guard does NOT separate; a negative (0.681) outscores
  a positive (0.522). It only "passes" because both positives happen to clear
  its 0.50 cut while being ANDed with the clear fact. It is not an independent
  discriminator on this corpus.
- Threshold-robust band (primary, confusion unchanged): **[0.73, 0.78], width
  0.05**. Move the shipped 0.75 by ±0.03 and the confusion breaks.
- Stratified bootstrap CI collapses to [1.0, 1.0] — but that is an ARTIFACT of
  having only 2 positives (resampling 2 points cannot express uncertainty), NOT
  evidence of robustness. The margin + band are the honest signal.
- Verdict: precision 1.0 is real on the corpus but rests on 2 positives, a
  0.052 margin, and a 0.05-wide threshold band. NOT statistically trustworthy;
  operator's diagnosis confirmed quantitatively.

### GRASP_OBJECT (n=10, pos=4, neg=6) — weak separation
- `grasp@eye` episode-mean: pos = [0.427, 0.508, 0.519, 0.669], neg =
  [0.269, 0.424, 0.467, 0.474, 0.495, 0.659] → margin = **-0.233** (a failure
  at 0.659 outscores 3 of 4 successes). Point precision 0.75 / recall 0.75.
- Bootstrap95: precision **[0.4, 1.0]**, recall [0.25, 1.0] — could be as bad as
  0.4. The eye-view grasp signal barely beats chance on this corpus.
- Threshold-robust band width 0.0 at the shipped 0.5 (any move changes confusion).

### Root blocker for the REAL fix (item #1: large balanced corpus)
The only way to make these numbers trustworthy is more POSITIVES per skill
(target ≥15 each). That requires the base/arm policy inference server
(~10.7 GB, `xiaomi-server`-class) to roll out CloseBlenderLid. The single RTX3090
on amp2 is occupied by the PROTECTED [1-1] exp4 GRPO trainer
(`xiaomi-grpo-trainer-t_3ed65912`, 간섭 금지): ~10.4–11.7 GB free, no safe
headroom for a second 10.7 GB server (logged: co-residence drops the trainer RPC
with struct.error). The ~15 extra rollouts in `rl-env/results` are all
not-even-grasped FAILURES — they add trivial negatives, zero new positives, so
they do not fix the positive-scarcity that the whole complaint is about.
Fresh collection (and the item-#4 end-to-end re-verification that depends on it)
is GPU-blocked until the trainer finishes or a GPU window is granted.

### Free (no-GPU, no-new-data) optimization space is EXHAUSTED
Swept every cached question × view × {mean,last5} on the existing corpus
(`calib_stats.py` companion sweep). Result: the shipped operating points are
already the best achievable on this data — there is no un-tried question/view
that widens the margin, so tuning cannot substitute for more positives.
- PLACE: `place_clear@right` (mean) is the ONLY (question,view) with a positive
  separation margin (+0.052, AUC 1.0). Every other combination has a NEGATIVE
  margin (place_combined@right -0.010, place_released@eye -0.091, seated@right
  -0.159, …). The shipped PLACE gate is already optimal for this corpus.
- GRASP: NO (question,view,aggregation) yields a positive margin. Best is
  grasp@eye mean (AUC 0.750, margin -0.233); grasp@left mean AUC 0.708 also
  negative margin. GRASP success/failure are NOT linearly separable to this VLM
  on the current 4-positive corpus regardless of view — a data problem.

Conclusion: further accuracy requires MORE POSITIVES (GPU collection), not more
tuning. Honest per operator's "성공 포장 금지".

## Iteration 4 — balanced 15+/15- corpus and re-calibration (task t_a309678b)

The positive-data blocker was removed by dedicating amp_csi's RTX3090 to
collection. The new seed-major corpus contains per-step 3-camera observations,
14-D proprio, sim predicates / success_step (offline labels only), and videos.
`build_balanced_corpus.py` selected exactly 15 success + 15 failure episodes for
each of GRASP, MOVE, and PLACE (90 skill episodes total). Raw eligible counts at
selection time were GRASP +15/-15, MOVE +18/-15, PLACE +15/-27.

Primary per-view cache pass: 10,182 forwards, 710.7 s, GPU peak 10.41 GB,
frame-stride 4. A separate full-concat baseline cache used 3,067 forwards and
265.6 s. All threshold/view/question/aggregation replay after caching was CPU
only. The temporal label is strict: a frame before `success_step` is negative;
an early ADVANCE counts as both FP and FN.

### What the larger dataset changed

The old 2-positive PLACE claim did not survive: the shipped right-view episode
mean gate fell to precision 0.60 / recall 0.20. The larger corpus therefore
confirmed the operator's complaint rather than validating the earlier result.

The new endpoint/window sweep found:

  skill   rule / aggregation                         precision  recall  TP FP FN TN
  GRASP   grasp@eye final-10 mean >= 0.45327             .667    .933  14  7  1  8
  MOVE    move@left final-5 mean >= 0.88857              .900    .600   9  1  6 14
  PLACE   combined@eye final frame >= 0.89973             .900    .600   9  1  6 14

PLACE and GRASP reporting gates were updated to these measured points. MOVE is
a waypoint and remains a diagnostic endpoint metric rather than a strict final
task gate. The cache-replayed shipped SuccessGate now measures:

  SUCCESS judge AFTER                 precision  recall  TP FP FN TN
  GRASP_OBJECT                            .667    .933  14  7  1  8
  PLACE_OBJECT                            .900    .600   9  1  6 14

  old full-concat latch BEFORE         precision  recall  TP FP FN TN
  GRASP_OBJECT                            .538    .933  14 12  1  3
  PLACE_OBJECT                            .400    .667  10 15  5  0

Thus PLACE false positives fall 15 -> 1 and precision rises .40 -> .90, with a
small recall cost (.667 -> .60). GRASP improves but still misses the .90
precision target; forcing precision >=.90 leaves recall <=.133, so the deployed
point uses the higher-F1 .667/.933 trade-off and is labelled as such.

### Boundary judgment remains the unsolved part

The strict online earliest-transition replay is poor even after sweeping every
question x view x tau x latch x hold combination:

  GRASP best precision .25 / recall .067 (frame AUC best .665)
  MOVE  best precision .40 / recall .133 (frame AUC best .712)
  PLACE best precision .333 / recall .067 (frame AUC best .896)

The same VLM can recognize a final completion checkpoint for MOVE/PLACE, but it
cannot reliably identify the earliest safe transition frame. This is not hidden
by the endpoint result. An end-to-end run may therefore still advance early or
time out; the strict endpoint gate improves honest reporting, not the online
boundary controller. Full JSON and manifests are in
`calib_out/balanced_t_a309678b/`.

### End-to-end re-check with the calibrated boundary config

Three fresh full-task seeds used the emitted per-skill boundary config
(GRASP eye/tau .80, MOVE left/.95, PLACE eye/.98) plus the new endpoint gates.
All execution decisions remained obs-only; sim predicates were read only after
the episode for evaluation.

  seed  sim task success  obs_task_success  outcome
   40        false             false         PLACE timeout twice
   41        false             true          residual strict false-positive
   42        false             false         PLACE timeout twice

Result: official success **0/3**, obs-vs-sim agreement **2/3**. This is worse
than the lenient uncalibrated boundary's 1/3 official success on the same seeds:
the high-precision temporal operating points trade early false alarms for PLACE
timeouts, and seed41 still produces one endpoint false-positive. Therefore the
requested "model-only architecture smoothly progresses" claim is **not
established**. The data-driven success reporter is materially better, but the
online boundary controller remains the blocking component. Traces, summaries,
and three overlay videos are stored under
`calib_out/balanced_t_a309678b/e2e_optimized/` and tracking Inference v3.

## Iteration 5 — causal sequence boundary model + uncensored dwell (task t_51f35471)

Iteration 4's corpus ended at the first GT-positive frame. This censored the
very signal an online boundary latch needs: only 5/15 GRASP, 4/15 MOVE, and
3/15 PLACE positives had any scored observation at or after the boundary, so
their maximum observable recalls were .333, .267, and .200 respectively.

The simulator replay collector therefore reproduced successful trajectories
and recorded 32 real post-success hold steps (camera observations + 14-D
proprio). GT predicates were used only to label the transition. Runtime inputs
remain the frozen VLM P(yes), the 14-D proprio state, and an 8-query causal
history. `train_sequence_boundary.py` uses a task-seeded, whole-rollout
stratified 18/6/6 train/calibration/test split, selects view/model/threshold/dwell
only on calibration, and exports a standard-library-only 72-D model consumed by
`ObsVLMVerifier`.

  skill   selected model                    held-out P/R  early FP  mean offset
  GRASP   left, logistic, tau .93, dwell 2      .667/.667       0       +3.0
  MOVE    right, ExtraTrees, tau .63, dwell 1   .750/1.00       0      +19.0
  PLACE   eye, ExtraTrees, tau .75, dwell 2     1.00/.333       0        0.0

Calibration met P>=.9/R>=.5 for every skill, but the untouched held-out split
did not: GRASP precision and PLACE recall are below target, and MOVE's +19-step
delay is large. The correct conclusion is a frozen-VLM representation limit,
not a threshold success. The sequence model removes all held-out early
transitions and improves recall beyond the censored-data ceiling for GRASP and
MOVE, but does not reliably separate every failed rollout or recover PLACE.

### Ten-seed full-task re-check

Seeds 40--49 were run with the exported sequence models and real combined
policy/VLM RPC server. Official simulator success was 6/10 (the policy-attainable
ceiling observed in this run), task-level obs/sim agreement was 5/10, and there
were zero task-level false-positive success reports. The strict obs success gate
reported only 1/10, producing five false negatives. Retry attempts timed out 4
times in GRASP, 5 in MOVE, and 11 in PLACE. Thus the requested >=.9 agreement is
not met; post-success sequence supervision fixes censoring but the frozen VLM
still cannot provide a sufficiently sensitive PLACE boundary/success signal.

The first RPC launch used Xiaomi's policy-only server and correctly stalled when
the verifier opened its second connection. The validation was restarted with
the repository's intended combined protocol (concurrent policy/VLM clients,
one CUDA lock, and `op=vlm_score` on one frozen model load). Full calibration,
ten summaries/traces, and ten videos are in
`calib_out/balanced_t_a309678b/{sequence_boundary_calibration.json,e2e_sequence/}`.
