#!/usr/bin/env python3
"""Extract CloseBlenderLid episodes from the RoboCasa365 LeRobot demos and inspect
trajectory content, to decide how to segment GRASP / MOVE / RELEASE.

Writes:
  results/closeblenderlid_episodes.json   per-episode index (row ranges, length, task)
  results/traj_probe.json                 stats on state/action/reward/done for a few eps
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
os.makedirs(OUT, exist_ok=True)


def dl(fn):
    return hf_hub_download(repo_id=REPO, filename=fn, repo_type="dataset", local_dir=CACHE)


def main():
    ep = pq.read_table(dl("meta/episodes/chunk-000/file-000.parquet")).to_pydict()
    nep = len(ep["episode_index"])
    # CloseBlenderLid episodes: source_prefix contains "CloseBlenderLid" (not Open)
    cbl = []
    for i in range(nep):
        sp = ep["source_prefix"][i]
        if "CloseBlenderLid" in sp:
            cbl.append({
                "episode_index": int(ep["episode_index"][i]),
                "length": int(ep["length"][i]),
                "from": int(ep["dataset_from_index"][i]),
                "to": int(ep["dataset_to_index"][i]),
                "data_chunk": int(ep["data/chunk_index"][i]),
                "data_file": int(ep["data/file_index"][i]),
                "source_prefix": sp,
                "source_episode_index": int(ep["source_episode_index"][i]),
                "tasks": ep["tasks"][i],
            })
    print("CloseBlenderLid episodes:", len(cbl))
    if cbl:
        lens = [e["length"] for e in cbl]
        print("length min/mean/max:", min(lens), round(sum(lens)/len(lens),1), max(lens))
        print("data files used:", sorted(set((e["data_chunk"], e["data_file"]) for e in cbl)))
        print("distinct task phrasings:", set(tuple(e["tasks"]) for e in cbl))
    json.dump(cbl, open(os.path.join(DATA_OUT, "closeblenderlid_episodes.json"), "w"), indent=2)

    # Load the needed data parquet files (only those containing CBL rows), extract a few eps.
    needed = sorted(set((e["data_chunk"], e["data_file"]) for e in cbl))
    tables = {}
    for (c, f) in needed:
        path = dl(f"data/chunk-{c:03d}/file-{f:03d}.parquet")
        tables[(c, f)] = pq.read_table(path)
        print("loaded data", (c, f), "rows", tables[(c, f)].num_rows)

    probe = {"episodes_probed": []}
    for e in cbl[:3]:
        tbl = tables[(e["data_chunk"], e["data_file"])]
        # rows for this episode: global index range from/to maps into the concatenated
        # dataset; within a single-file dataset the row index == global index, but v3
        # splits across files. Use episode_index filter via the 'episode_index' column.
        ei_col = tbl.column("episode_index").to_numpy()
        mask = ei_col == e["episode_index"]
        n = int(mask.sum())
        sub = tbl.filter(mask)
        state = np.array(sub.column("observation.state").to_pylist(), dtype=float)
        action = np.array(sub.column("action").to_pylist(), dtype=float)
        reward = np.array(sub.column("next.reward").to_pylist(), dtype=float).reshape(-1)
        done = np.array(sub.column("next.done").to_pylist()).reshape(-1)
        gripper_state = state[:, -2:] if state.shape[1] >= 2 else None
        rec = {
            "episode_index": e["episode_index"], "n_rows_in_file": n, "meta_length": e["length"],
            "state_shape": list(state.shape), "action_shape": list(action.shape),
            "state_first": state[0].tolist(), "state_last": state[-1].tolist(),
            "state_min": state.min(0).tolist(), "state_max": state.max(0).tolist(),
            "action_first": action[0].tolist(), "action_last": action[-1].tolist(),
            "action_min": action.min(0).tolist(), "action_max": action.max(0).tolist(),
            "action_dim11_gripper_head": action[:8, -1].tolist(),
            "action_dim11_gripper_tail": action[-8:, -1].tolist(),
            "reward_nonzero_idx": np.nonzero(reward)[0].tolist(),
            "reward_sum": float(reward.sum()), "reward_last10": reward[-10:].tolist(),
            "done_true_idx": np.nonzero(done)[0].tolist(),
        }
        probe["episodes_probed"].append(rec)
        print("\nep", e["episode_index"], "len", n, "state", state.shape, "action", action.shape,
              "reward_nonzero", rec["reward_nonzero_idx"][:5], "done_idx", rec["done_true_idx"])
        print("  state_last (16d):", [round(x,3) for x in state[-1]])
        print("  action_last (12d):", [round(x,3) for x in action[-1]])
    json.dump(probe, open(os.path.join(OUT, "traj_probe.json"), "w"), indent=2)
    print("\nwrote closeblenderlid_episodes.json + traj_probe.json")


if __name__ == "__main__":
    main()
