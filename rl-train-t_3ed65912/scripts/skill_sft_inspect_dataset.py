#!/usr/bin/env python3
"""Inspect the public RoboCasa365 CloseBlenderLid LeRobot demos (v3.0).

Dataset: ember-lab-berkeley/robocasa365-pretrain-atomic (PUBLIC, no token).
Goal-1 groundwork: figure out (a) which task_index is CloseBlenderLid, (b) how
many episodes / frames it has, (c) what observation.state[16] and action[12]
actually contain, and (d) whether next.reward / next.done give a usable
success/termination signal for skill segmentation.

Runs CPU-only inside an ephemeral xiaomi-cu121 container (pyarrow pip-installed).
Downloads only meta + the parquet chunk(s); no GPU touched.
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "ember-lab-berkeley/robocasa365-pretrain-atomic"
OUT = os.environ.get("OUT", "/out")
os.makedirs(OUT, exist_ok=True)


def dl(fn):
    return hf_hub_download(repo_id=REPO, filename=fn, repo_type="dataset",
                           local_dir=os.environ.get("SFT_HF_CACHE", os.path.join(OUT, "hf_cache")))


def main():
    info = json.load(open(dl("meta/info.json")))
    print("codebase", info["codebase_version"], "eps", info["total_episodes"],
          "frames", info["total_frames"], "tasks", info["total_tasks"], "fps", info["fps"])

    # tasks.parquet: maps task_index -> natural-language task string
    tasks_tbl = pq.read_table(dl("meta/tasks.parquet")).to_pydict()
    print("\n=== tasks.parquet columns:", list(tasks_tbl.keys()))
    # find CloseBlenderLid rows
    # column layout unknown; dump first rows
    ncol = len(next(iter(tasks_tbl.values())))
    print("num task rows:", ncol)
    # locate the text column
    text_col = None
    for k, v in tasks_tbl.items():
        if v and isinstance(v[0], str):
            text_col = k
            break
    idx_col = None
    for k, v in tasks_tbl.items():
        if k != text_col:
            idx_col = k
    blender_rows = []
    for i in range(ncol):
        txt = tasks_tbl[text_col][i]
        if "blender" in str(txt).lower() and "lid" in str(txt).lower():
            row = {k: tasks_tbl[k][i] for k in tasks_tbl}
            blender_rows.append(row)
    print("\n=== blender-lid task rows (text col=%s idx col=%s) ===" % (text_col, idx_col))
    for r in blender_rows[:40]:
        print(r)

    # episodes meta
    ep_tbl = pq.read_table(dl("meta/episodes/chunk-000/file-000.parquet")).to_pydict()
    print("\n=== episodes.parquet columns:", list(ep_tbl.keys()))
    nep = len(next(iter(ep_tbl.values())))
    print("num episode rows:", nep, "| first row:",
          {k: ep_tbl[k][0] for k in ep_tbl})
    json.dump({"info": info, "tasks_columns": list(tasks_tbl.keys()),
               "blender_task_rows": [{k: (v if not isinstance(v, (np.integer,)) else int(v))
                                      for k, v in r.items()} for r in blender_rows],
               "episodes_columns": list(ep_tbl.keys()),
               "episodes_first_row": {k: (ep_tbl[k][0].tolist() if hasattr(ep_tbl[k][0], "tolist") else ep_tbl[k][0]) for k in ep_tbl}},
              open(os.path.join(OUT, "dataset_meta_summary.json"), "w"),
              indent=2, default=str)
    print("\nwrote", os.path.join(OUT, "dataset_meta_summary.json"))


if __name__ == "__main__":
    main()
