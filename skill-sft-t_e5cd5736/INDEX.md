# INDEX — Xiaomi RoboCasa365 skill-conditioned VLA: SFT → adapter compare → RL (t_e5cd5736)

Per-skill (GRASP_HANDLE / MOVE_LID_TO_CLOSED / RELEASE_HANDLE) supervised
flow-matching SFT stage feeding the existing skill-conditioned RL pipeline. Reuses
`rl-env-t_4f3f2b20` and `rl-train-t_3ed65912` by import only (no duplicate project);
pinned model revision `3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4`.

## Status (phase-by-phase)

| Goal | What | State |
|------|------|-------|
| 1 | Acquire + skill-segment RoboCasa365 CloseBlenderLid demos; split + mask + manifest | **DONE (GPU-free)** |
| 2 | Conditioning schema (goal + skill_id + NL + object/dest/constraints) → policy | **DONE (GPU-free)** |
| 3 | Per-skill DiT LoRA CFM SFT; gates: eta=0 → tiny-overfit → heldout → entry-state rollout | design DONE; run GATED on GPU |
| 4 | Same-seed baseline vs per-skill eval (success, wrong-skill, post-boundary, latency, Δaction) + videos | GATED on GPU + Goal 3 |
| 5 | Equal-budget compare NL-only vs NL+skill-id vs shared-LoRA+learned-embedding; select by heldout | GATED on GPU + Goal 3 |
| 6 | Verified multi-epoch GRPO from selected SFT ckpt, GRASP→MOVE→RELEASE; RECAP feasibility | GATED on GPU + SFT gate |

**GPU sequencing:** Goals 3–6 need the v4 GPU (checkpoint forward / sim rollout).
They are held until the GR00T comparison eval (`groot-eval:t_ff372eea`) frees the
GPU — do not intrude on its run. Goals 1–2 used only ephemeral CPU containers.

## Goal 1 — data (PASS)

- Source: `ember-lab-berkeley/robocasa365-pretrain-atomic` (PUBLIC LeRobot v3.0,
  matches the checkpoint's 16-D state / 12-D action). **106 CloseBlenderLid demos**,
  all official-success. **RELEASE data present** (105/106 issue a release command).
- `observation.state` has NO object pose ⇒ segment on policy-visible signals
  (gripper command square wave + finger qpos audit + terminal reward), not sim
  predicates (which would need GPU/EGL replay; a sample cross-check is deferred).
- Split 74/16/16 (trajectory-level, seed 20260915), per-skill action loss mask,
  manifest with SHA-256 of every source file.
- Details: `docs/DATA_FINDINGS.md`. Artifacts: `results/{data_manifest,skill_segments,
  closeblenderlid_episodes,dataset_meta_summary,traj_probe,segmentation_probe,finger_timeline}.json`.

## Goal 2 — conditioning (PASS)

- Policy's only text slot is `instruction` in `client.infer(...)`. Schema maps overall
  goal + stable skill_id (0/1/2) + NL + object/destination/constraints onto the three
  ablation arms. `configs/conditioning_schema.json`, `src/conditioning.py`,
  `tests/test_conditioning.py` (6/6 pass locally).

## Goal 3 — SFT design (ready to run)

CFM loss `||v_θ((1-t)x0+t·x1, t) - (x1-x0)||²` on the masked skill span; per-skill
LoRA via existing `PerSkillLoRALinear`; checkpoint carries optimizer/RNG/config/
data-hash/git+model-rev. Four-gate verification before RL. `docs/SFT_DESIGN.md`.

## Reproduce (GPU-free parts, on v4 via ephemeral CPU container)

```
cd /home/v4/skill-sft-t_e5cd5736
docker run --rm --cpus 4 -v $PWD:/work -e OUT=/work/results --entrypoint bash \
  xiaomi-cu121:t_9f03a613 -lc \
  "python3 -m pip install --quiet pyarrow; python3 /work/scripts/segment_and_manifest.py"
python3 src/conditioning.py && python3 tests/test_conditioning.py   # local, stdlib only
```
