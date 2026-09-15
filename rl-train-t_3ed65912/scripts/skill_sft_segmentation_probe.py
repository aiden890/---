#!/usr/bin/env python3
"""Probe whether GRASP / MOVE_HOLDING / RELEASE are separable from the RECORDED
proprioception + gripper + reward signal alone (no object pose in the dataset).

For each of several CloseBlenderLid demos, dump the time series of:
  - gripper finger state  (observation.state last 2 dims)
  - gripper action command (action last dim)
  - EE z height           (best-guess EE-position dim in state)
  - next.reward           (terminal success onset)
so we can decide the segmentation method and check if a RELEASE (gripper reopen
after success) phase is present in the data.
"""
import json
import os

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "ember-lab-berkeley/robocasa365-pretrain-atomic"
OUT = os.environ.get("OUT", "/out")
DATA_OUT = os.environ.get("DATA_OUT", OUT)  # durable data artifacts (manifest/segments/episode index)
CACHE = os.environ.get("SFT_HF_CACHE", os.path.join(OUT, "hf_cache"))


def dl(fn):
    return hf_hub_download(repo_id=REPO, filename=fn, repo_type="dataset", local_dir=CACHE)


def main():
    cbl = json.load(open(os.path.join(DATA_OUT, "closeblenderlid_episodes.json")))
    tbl = pq.read_table(dl("data/chunk-000/file-000.parquet"))
    ei = tbl.column("episode_index").to_numpy()

    summary = []
    for e in cbl[:8]:
        sub = tbl.filter(ei == e["episode_index"])
        state = np.array(sub.column("observation.state").to_pylist(), dtype=float)
        action = np.array(sub.column("action").to_pylist(), dtype=float)
        reward = np.array(sub.column("next.reward").to_pylist(), dtype=float).reshape(-1)
        n = len(state)
        grip_fingers = state[:, -2:]              # last 2 state dims
        grip_cmd = action[:, -1]                  # last action dim (gripper)
        # candidate EE z: state has base(0:3?) + ... ; probe each dim's variance
        # We know from eval entry EE-first 14D; here state is 16D. Report per-dim range.
        rng = (state.max(0) - state.min(0))
        # gripper finger opening = fingers[:,0]-fingers[:,1] roughly; detect closed periods
        opening = grip_fingers[:, 0] - grip_fingers[:, 1]
        # normalize gripper command sign timeline: -1 close, +1 open (robosuite convention)
        close_mask = grip_cmd < -0.5
        open_mask = grip_cmd > 0.5
        # find first sustained close (>=10 steps) and any reopen after it
        def first_sustained(mask, k=10):
            c = 0
            for i, m in enumerate(mask):
                c = c + 1 if m else 0
                if c >= k:
                    return i - k + 1
            return None
        grasp_close_start = first_sustained(close_mask, 10)
        reward_on = int(np.nonzero(reward)[0][0]) if reward.any() else None
        # reopen after reward: gripper command goes open in the last quarter
        reopen_after = None
        if reward_on is not None:
            tail = open_mask[reward_on:]
            if tail.any():
                reopen_after = reward_on + int(np.nonzero(tail)[0][0])
        rec = {
            "episode_index": e["episode_index"], "len": n,
            "grip_cmd_head8": [round(x, 2) for x in grip_cmd[:8]],
            "grip_cmd_tail12": [round(x, 2) for x in grip_cmd[-12:]],
            "n_close_cmd": int(close_mask.sum()), "n_open_cmd": int(open_mask.sum()),
            "grasp_close_start(sustained10)": grasp_close_start,
            "reward_onset": reward_on,
            "reopen_cmd_after_reward": reopen_after,
            "finger_opening_first": round(float(opening[0]), 4),
            "finger_opening_at_grasp": (round(float(opening[grasp_close_start]), 4)
                                        if grasp_close_start is not None else None),
            "finger_opening_last": round(float(opening[-1]), 4),
            "state_perdim_range": [round(x, 3) for x in rng],
        }
        summary.append(rec)
        print(f"\nep {e['episode_index']} len {n}: grasp_close@{grasp_close_start} "
              f"reward_on@{reward_on} reopen_after@{reopen_after} "
              f"close_cmds={int(close_mask.sum())} open_cmds={int(open_mask.sum())}")
        print("  grip_cmd tail:", rec["grip_cmd_tail12"])
        print("  finger opening first/grasp/last:", rec["finger_opening_first"],
              rec["finger_opening_at_grasp"], rec["finger_opening_last"])
    json.dump(summary, open(os.path.join(OUT, "segmentation_probe.json"), "w"), indent=2)
    # aggregate: does a reopen-after-reward (RELEASE) exist in the demos?
    n_reopen = sum(1 for r in summary if r["reopen_cmd_after_reward"] is not None)
    print(f"\n=== RELEASE presence: {n_reopen}/{len(summary)} probed demos show gripper "
          f"reopen command after reward onset ===")


if __name__ == "__main__":
    main()
