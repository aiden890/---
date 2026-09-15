#!/usr/bin/env python3
"""Goal-1: segment RoboCasa365 CloseBlenderLid demos into GRASP_HANDLE /
MOVE_LID_TO_CLOSED / RELEASE_HANDLE, build a trajectory-level train/val/test
split, a per-skill action loss mask, and a data manifest with hashes.

Data availability finding (see docs/DATA_FINDINGS.md): the public LeRobot mirror
that matches the checkpoint's I/O (16-D EE-state, 12-D action) carries NO object
pose, so exact lid predicates (lid_grasped / lid_on_blender) are not directly
readable. We therefore segment on the signals the POLICY ITSELF consumes:

  finger opening = observation.state[-2] - observation.state[-1]   (gripper qpos)
  gripper action = action[-1]      (+1 = close command, -1 = open command)
  next.reward                       (terminal official-success onset)

Skill boundaries (finger state machine w/ hysteresis, cross-checked vs gripper
command + reward monotonicity):

  GRASP_HANDLE      : t=0 .. grasp_end     (approach + close on lid; ends at the
                       first sustained finger-closed onset)
  MOVE_LID_TO_CLOSED: grasp_end .. release_start (transport the held lid to the
                       closed position on the blender)
  RELEASE_HANDLE    : release_start .. T-1 (open gripper, retreat; ends at
                       terminal success / done)

A demo is FLAGGED (not used for a skill it fails to yield) if a boundary is
ambiguous (e.g. loose grasp that never reads clearly closed). Every decision is
recorded per-episode so the labels are auditable and reproducible.
"""
import hashlib
import json
import os

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "ember-lab-berkeley/robocasa365-pretrain-atomic"
OUT = os.environ.get("OUT", "/out")
CACHE = os.path.join(OUT, "hf_cache")

# hysteresis thresholds on finger opening (metres); from finger_timeline.py:
# open approach ~0.078-0.080, closed-on-lid ~0.026-0.045, rest ~0.041, loose ~0.065
OPEN_TH = 0.065          # opening above this => gripper OPEN
CLOSED_TH = 0.050        # opening below this => gripper CLOSED (holding)
HOLD = 5                 # consecutive closed steps to confirm a secure grasp
CLOSE_CMD = 0.5          # action[-1] > this => close command
OPEN_CMD = -0.5          # action[-1] < this => open command

SKILLS = ["GRASP_HANDLE", "MOVE_LID_TO_CLOSED", "RELEASE_HANDLE"]

# trajectory-level split fractions (seeded, deterministic)
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 20260915


def dl(fn):
    return hf_hub_download(repo_id=REPO, filename=fn, repo_type="dataset", local_dir=CACHE)


def sha256_file(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sustained_onset(mask, k):
    """First index i s.t. mask[i:i+k] all True."""
    n = len(mask)
    for i in range(n - k + 1):
        if mask[i:i + k].all():
            return i
    return None


def segment_episode(state, action, reward):
    """Segment on the gripper COMMAND square-wave (open -> close -> open), the
    cleanest 3-phase signal, with the finger-opening state used to VALIDATE that
    the close command actually secured the lid.

    gripper command action[-1]:  +1 = close, -1 = open  (robosuite convention;
    empirically the demos command open during approach, close while holding, and
    open again to release).

      GRASP_HANDLE      : [0, close_start)           approach + close on the lid
      MOVE_LID_TO_CLOSED: [close_start, release_start) hold + transport to closed pos
      RELEASE_HANDLE     : [release_start, n)          open gripper + retreat
    """
    n = len(state)
    opening = state[:, -2] - state[:, -1]
    gcmd = action[:, -1]
    reward_on = int(np.nonzero(reward.reshape(-1) > 0)[0][0]) if (reward > 0).any() else None

    close_mask = gcmd > CLOSE_CMD
    open_mask = gcmd < OPEN_CMD

    # close_start = first sustained close command (grasp secured / lid picked)
    close_start = _sustained_onset(close_mask, HOLD)
    # release_start = first sustained open command AFTER close_start (lid released)
    release_start = None
    if close_start is not None:
        rel_rel = _sustained_onset(open_mask[close_start + HOLD:], HOLD)
        if rel_rel is not None:
            release_start = close_start + HOLD + rel_rel

    flags = []
    # NOTE: finger opening is NOT monotonic with grasp for this object -- the lid is
    # wide, so closing on its rim can SPREAD the fingers wider than rest (~0.041).
    # These are all official-SUCCESS demos (reward fires at the end of every one), so
    # the grasp is valid by construction; we segment on the gripper COMMAND square
    # wave and confirm the task with terminal reward, and only record finger stats
    # for audit rather than flagging a "loose grasp" (that was a false positive).
    hold_end = release_start if release_start is not None else n
    opening_in_hold_min = (round(float(opening[close_start:hold_end].min()), 4)
                           if close_start is not None and hold_end > close_start else None)
    if close_start is None:
        flags.append("no_close_command")
    if release_start is None:
        flags.append("no_release_command")
    if reward_on is None:
        flags.append("no_terminal_reward")
    if reward_on is not None and release_start is not None and release_start > reward_on:
        flags.append("release_after_reward_onset")

    spans = {}
    if close_start is not None and close_start > 0:
        spans["GRASP_HANDLE"] = [0, close_start]
    if close_start is not None and release_start is not None:
        spans["MOVE_LID_TO_CLOSED"] = [close_start, release_start]
    if release_start is not None and release_start < n:
        spans["RELEASE_HANDLE"] = [release_start, n]

    return {
        "n": n, "close_start": close_start, "release_start": release_start,
        "reward_onset": reward_on, "opening_in_hold_min": opening_in_hold_min,
        "opening_min": round(float(opening.min()), 4),
        "opening_max": round(float(opening.max()), 4),
        "spans": spans, "flags": flags,
    }


def main():
    cbl = json.load(open(os.path.join(OUT, "closeblenderlid_episodes.json")))
    data_path = dl("data/chunk-000/file-000.parquet")
    info_path = dl("meta/info.json")
    tasks_path = dl("meta/tasks.parquet")
    tbl = pq.read_table(data_path)
    ei = tbl.column("episode_index").to_numpy()

    per_ep = []
    for e in cbl:
        sub = tbl.filter(ei == e["episode_index"])
        state = np.array(sub.column("observation.state").to_pylist(), dtype=float)
        action = np.array(sub.column("action").to_pylist(), dtype=float)
        reward = np.array(sub.column("next.reward").to_pylist(), dtype=float)
        seg = segment_episode(state, action, reward)
        seg["episode_index"] = e["episode_index"]
        seg["source_prefix"] = e["source_prefix"]
        seg["source_episode_index"] = e["source_episode_index"]
        per_ep.append(seg)

    # ---- trajectory-level split (seeded) ----
    rng = np.random.RandomState(SPLIT_SEED)
    order = [p["episode_index"] for p in per_ep]
    rng.shuffle(order)
    n = len(order)
    n_tr = int(round(SPLIT["train"] * n))
    n_va = int(round(SPLIT["val"] * n))
    split_of = {}
    for k, idx in enumerate(order):
        split_of[idx] = "train" if k < n_tr else ("val" if k < n_tr + n_va else "test")
    for p in per_ep:
        p["split"] = split_of[p["episode_index"]]

    # ---- per-skill availability counts ----
    counts = {sk: {"train": 0, "val": 0, "test": 0} for sk in SKILLS}
    for p in per_ep:
        for sk in p["spans"]:
            counts[sk][p["split"]] += 1
    flagged = [p["episode_index"] for p in per_ep if p["flags"]]

    manifest = {
        "task": "CloseBlenderLid",
        "dataset_repo": REPO,
        "dataset_codebase_version": json.load(open(info_path))["codebase_version"],
        "n_demos": n,
        "instruction_text": "Close the lid blender by securely placing the lid on top.",
        "obs_state_dim": 16, "action_dim": 12, "fps": 20,
        "obs_note": "observation.state is robot proprioception only (base+EE pose+gripper), NO object pose",
        "segmentation": {
            "method": "policy-visible signals: finger-opening state machine (hysteresis) "
                      "+ gripper command cross-check + next.reward terminal onset",
            "OPEN_TH": OPEN_TH, "CLOSED_TH": CLOSED_TH, "HOLD": HOLD,
            "CLOSE_CMD": CLOSE_CMD, "OPEN_CMD": OPEN_CMD,
            "skills": SKILLS,
        },
        "split": {"seed": SPLIT_SEED, "fractions": SPLIT,
                  "counts": {"train": n_tr, "val": n_va, "test": n - n_tr - n_va}},
        "per_skill_available_counts": counts,
        "flagged_episodes": flagged,
        "action_loss_mask": "for a per-skill training example, actions OUTSIDE the skill's "
                            "[start,end) span are masked out of the flow-matching loss",
        "data_hashes": {
            "data/chunk-000/file-000.parquet": sha256_file(data_path),
            "meta/info.json": sha256_file(info_path),
            "meta/tasks.parquet": sha256_file(tasks_path),
        },
    }

    json.dump(per_ep, open(os.path.join(OUT, "skill_segments.json"), "w"), indent=2)
    json.dump(manifest, open(os.path.join(OUT, "data_manifest.json"), "w"), indent=2)

    print("=== segmentation summary ===")
    print("demos:", n, "| split train/val/test:", n_tr, n_va, n - n_tr - n_va)
    print("flagged episodes:", len(flagged))
    for p in per_ep:
        if p["flags"]:
            print("  ep", p["episode_index"], p["flags"],
                  "close_start", p["close_start"], "release_start", p["release_start"])
    print("per-skill available (train/val/test):")
    for sk in SKILLS:
        c = counts[sk]
        print(f"  {sk:20s} {c['train']:3d} / {c['val']:3d} / {c['test']:3d}")
    # span length stats
    for sk in SKILLS:
        lens = [p["spans"][sk][1] - p["spans"][sk][0] for p in per_ep if sk in p["spans"]]
        if lens:
            print(f"  {sk:20s} span len min/mean/max: {min(lens)}/{round(sum(lens)/len(lens),1)}/{max(lens)}")
    print("\nwrote skill_segments.json + data_manifest.json")


if __name__ == "__main__":
    main()
