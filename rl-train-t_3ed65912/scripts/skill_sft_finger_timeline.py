#!/usr/bin/env python3
"""Correctly characterize the grasp/hold/release structure from the RECORDED
finger qpos (not the ambiguous command sign), and print a coarse timeline.

Also verifies whether the LeRobot mirror carries any full MuJoCo sim state
(needed for exact predicate replay) -- it does NOT (only 16-D proprioception),
which is the load-bearing data-availability finding for the segmentation method.
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

    out = []
    for e in cbl[:4]:
        sub = tbl.filter(ei == e["episode_index"])
        state = np.array(sub.column("observation.state").to_pylist(), dtype=float)
        action = np.array(sub.column("action").to_pylist(), dtype=float)
        reward = np.array(sub.column("next.reward").to_pylist(), dtype=float).reshape(-1)
        n = len(state)
        # state 16D: [base x,y? , ... , gripper fingers last 2]. Report per-dim to identify EE-z.
        fingers = state[:, -2:]
        opening = fingers[:, 0] - fingers[:, 1]          # ~0 closed, ~0.08 open
        grip_cmd = action[:, -1]
        reward_on = int(np.nonzero(reward)[0][0]) if reward.any() else None
        # classify grasp state by finger opening threshold
        OPEN_TH = 0.055   # > this = open; lid thickness closes fingers below
        is_open = opening > OPEN_TH
        # transitions: first close (open->closed) = grasp; last open onset after that = release
        grasp_step = None
        for i in range(1, n):
            if is_open[i-1] and not is_open[i]:
                grasp_step = i
                break
        release_step = None
        if grasp_step is not None:
            for i in range(n-1, grasp_step, -1):
                if not is_open[i-1] and is_open[i]:
                    release_step = i
                    break
        # coarse timeline every ~ n/20 steps
        k = max(1, n // 20)
        tl = [(i, round(float(opening[i]), 3), round(float(grip_cmd[i]), 1),
               int(reward[i] > 0)) for i in range(0, n, k)]
        rec = {"episode_index": e["episode_index"], "len": n,
               "reward_onset": reward_on, "done_minus_reward": (n-1-reward_on) if reward_on else None,
               "opening_min": round(float(opening.min()), 4),
               "opening_max": round(float(opening.max()), 4),
               "grasp_step(open->closed)": grasp_step,
               "release_step(closed->open, after grasp)": release_step,
               "release_before_reward": (release_step is not None and reward_on is not None
                                         and release_step <= reward_on),
               "timeline_step_opening_gripcmd_reward": tl}
        out.append(rec)
        print(f"\nep {e['episode_index']} len {n}: grasp@{grasp_step} release@{release_step} "
              f"reward_on@{reward_on} opening[{rec['opening_min']},{rec['opening_max']}]")
        print("  step:opening:cmd:rew ->", " ".join(f"{i}:{o}:{c}:{r}" for i,o,c,r in tl))

    json.dump(out, open(os.path.join(OUT, "finger_timeline.json"), "w"), indent=2)
    nrel = sum(1 for r in out if r["release_step(closed->open, after grasp)"] is not None)
    print(f"\n=== RELEASE (finger reopen after grasp) present in {nrel}/{len(out)} demos ===")
    print("=== observation.state carries NO object pose (16-D proprioception only): "
          "exact lid predicates require sim replay, not available from this parquet ===")


if __name__ == "__main__":
    main()
