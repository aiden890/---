#!/usr/bin/env python3
"""Smoke test: verify video decode + frame-count alignment for one CloseBlenderLid demo,
so the SFT obs windows read the correct frames. CPU-only, ephemeral container."""
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "ember-lab-berkeley/robocasa365-pretrain-atomic"
CACHE = os.environ.get("SFT_HF_CACHE", "/tmp/hf/cache")
DATA_DIR = os.environ.get("DATA_DIR", "/train/data/skill_sft")
CAMS = ("observation.images.robot0_agentview_left",
        "observation.images.robot0_agentview_right",
        "observation.images.robot0_eye_in_hand")


def dl(fn):
    return hf_hub_download(repo_id=REPO, filename=fn, repo_type="dataset", local_dir=CACHE)


def main():
    import torchvision
    eps = json.load(open(os.path.join(DATA_DIR, "closeblenderlid_episodes.json")))
    ep_tbl = pq.read_table(dl("meta/episodes/chunk-000/file-000.parquet")).to_pydict()
    meta = {int(ep_tbl["episode_index"][i]): {k: ep_tbl[k][i] for k in ep_tbl}
            for i in range(len(ep_tbl["episode_index"]))}
    e = eps[0]
    ei = e["episode_index"]
    m = meta[ei]
    print("episode", ei, "meta length", e["length"])
    for cam in CAMS:
        ck, fi = m[f"videos/{cam}/chunk_index"], m[f"videos/{cam}/file_index"]
        t0, t1 = float(m[f"videos/{cam}/from_timestamp"]), float(m[f"videos/{cam}/to_timestamp"])
        path = dl(f"videos/{cam}/chunk-{ck:03d}/file-{fi:03d}.mp4")
        vid, _, info = torchvision.io.read_video(path, start_pts=t0, end_pts=t1,
                                                 pts_unit="sec", output_format="THWC")
        print(f"  {cam.split('.')[-1]:22s} frames={vid.shape} t=[{t0:.2f},{t1:.2f}] "
              f"fps={info.get('video_fps')} vs meta_len={e['length']}")
    print("VIDEO_DECODE_OK")


if __name__ == "__main__":
    main()
