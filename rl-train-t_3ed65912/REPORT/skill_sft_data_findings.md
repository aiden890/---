# Data findings — RoboCasa365 CloseBlenderLid skill segmentation (Goal 1)

Task t_e5cd5736, phase 1. All numbers reproduced on v4 in an ephemeral CPU
container (`docker run --rm --cpus 4 xiaomi-cu121:t_9f03a613`, no GPU touched,
GR00T eval left running). Scripts `scripts/skill_sft_*.py`; durable data under `data/skill_sft/`, evidence under `results/skill_sft/`.

## Source data (PASS — data exists, RELEASE present)

- Checkpoint I/O is 16-D `observation.state` + 12-D `action`. The public LeRobot
  mirror that matches this exactly is **`ember-lab-berkeley/robocasa365-pretrain-atomic`**
  (HF dataset, PUBLIC, no token, codebase v3.0). This is the LeRobot form of the
  RoboCasa365 pretrain-human atomic demos (`v1.0/pretrain/atomic/<Task>/…/lerobot`
  in `robocasa/utils/dataset_registry.py`).
- **CloseBlenderLid = 106 demonstrations** (task_index 8 NL / 9 canonical). Single
  NL phrasing in the data: *"Close the lid blender by securely placing the lid on
  top."* Episode length min/mean/max = 237 / 348 / 637 steps @ 20 fps. Cameras:
  3×256² h264 (`robot0_agentview_left` / `_right` / `_eye_in_hand`). All 106 carry a
  `next.reward` that fires near the end ⇒ all are **official-success** trajectories.
- **RELEASE_HANDLE data IS present** (the task brief asked to report if it were
  missing). 105/106 demos issue an explicit gripper-open (release) command after
  the hold; only ep 359 holds the closed lid through terminal success without a
  release command (a legitimate variant — its GRASP span is still usable).
  ⇒ Proceeding with all three skills, not just GRASP/MOVE.

## Key constraint: no object pose in the recorded state

`observation.state[16]` is **robot proprioception only** (mobile base pose + torso,
end-effector pose, and 2 gripper-finger qpos). It carries **no lid/blender pose**.
So the exact RoboCasa predicates used elsewhere in this project (`lid_grasped`,
`lid_on_blender`, `lid_upright_7deg`, `gripper_lid_far`) — which read MuJoCo object
bodies via `skill_eval.Sim.predicates()` — **cannot be computed from this parquet**,
and the LeRobot mirror does not ship MuJoCo sim state for replay.

Decision: segment on the signals the **policy itself consumes**, which cleanly
expose the 3-phase structure, and confirm the whole trajectory with terminal reward:

- gripper **command** `action[-1]`: **+1 = close, −1 = open** (robosuite convention;
  empirically open during approach → close while holding → open to release). This
  square wave is the cleanest boundary signal.
- gripper **finger qpos** `state[-2]-state[-1]` (opening): used only for audit. NB it
  is **not monotonic with grasp** for this object — the lid is wide, so closing on
  its rim can spread the fingers *wider* than rest (~0.041 m). An early "fingers
  never closed ⇒ loose grasp" validator was a **false positive** on 31 demos and was
  removed; grasp validity comes from the fact that every demo reaches official
  success.
- `next.reward`: terminal official-success onset (used to confirm, and to order the
  release relative to success).

## Segmentation (proxy-predicate labels)

Boundaries (all `[start, end)` in source-trajectory steps), from the gripper command
square wave with `HOLD=5` sustained-onset debouncing:

| skill                | span                          | meaning                                   |
|----------------------|-------------------------------|-------------------------------------------|
| `GRASP_HANDLE`       | `[0, close_start)`            | approach + close on the lid handle        |
| `MOVE_LID_TO_CLOSED` | `[close_start, release_start)`| hold + transport lid to the closed pos    |
| `RELEASE_HANDLE`     | `[release_start, n)`          | open gripper, retreat, settle to success  |

Span length min/mean/max (steps): GRASP 70/117/203 · MOVE 109/179/404 · RELEASE 38/53/120.

These are **proxy** labels (from policy-visible signals), not sim-replayed geometric
predicates. A sample-level cross-check against `skill_eval.Sim.predicates()` requires
re-running the demos' initial states in the simulator (GPU/EGL) and is deferred to the
GPU phase; the boundaries are unlikely to move much because the gripper command is a
direct expression of the demonstrator's grasp/release intent.

## Split, mask, manifest

- **Trajectory-level** train/val/test split = 74 / 16 / 16 demos, seeded
  (`SPLIT_SEED=20260915`), so no source trajectory leaks across splits.
- **Action loss mask**: for a per-skill training example, action steps *outside* the
  skill's `[start, end)` span are masked out of the flow-matching loss (spec recorded
  in the manifest; applied by the dataloader in the SFT phase).
- **Manifest** `data/skill_sft/data_manifest.json` records: repo + codebase version, demo
  count, instruction text, obs/action dims, the segmentation method + thresholds, the
  split (seed/fractions/counts), per-skill availability counts, flagged episodes, the
  loss-mask rule, and **SHA-256 of every source file** (data parquet + info + tasks).

Per-skill availability (train/val/test): GRASP 74/16/16 · MOVE 74/15/16 · RELEASE
74/15/16 (the two missing val/test entries are ep 359's absent MOVE/RELEASE).

## Artifacts

- `data/skill_sft/data_manifest.json` — the canonical manifest (hashes, split, method).
- `data/skill_sft/skill_segments.json` — per-episode spans, boundaries, reward onset, flags.
- `data/skill_sft/closeblenderlid_episodes.json` — 106 episode index (row ranges, lengths).
- `results/skill_sft/{dataset_meta_summary,traj_probe,segmentation_probe,finger_timeline}.json` — inspection evidence behind the decisions above.
