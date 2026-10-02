"""Audit public BC masks, source alignment and native normalization compatibility."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from online_bc.data.build_public_grasp import native_actions, native_state


def validate(root, source):
    import pyarrow.parquet as pq

    report = json.loads((root / "dataset.json").read_text())
    assert report["complete"] is True
    episodes = samples = 0
    ids = set()
    seeds = set()
    max_deadband = 0.0
    datasets = {}
    for path in sorted(root.glob("*/bc/manifest.json")):
        m = json.loads(path.read_text())
        proof = m["native_grasp_validation"]
        end = m["segments"][0]["end"]
        assert m["dataset_role"] == "expert_grasp_base" and proof["full_success"]
        assert (
            proof["native_predicate"]["bilateral_grasp"]
            and proof["native_predicate"]["mug_motion"] > 0.10
        )
        assert m["episode_id"] not in ids and m["seed"] not in seeds
        assert not 992001 <= m["seed"] <= 992030
        ids.add(m["episode_id"])
        seeds.add(m["seed"])
        if proof["annotated_end"] is not None:
            assert end == proof["annotated_end"]
        p = (
            source
            / m["source_dataset"]
            / "lerobot"
            / f"data/chunk-{m['source_episode_index'] // 1000:03d}/episode_{m['source_episode_index']:06d}.parquet"
        )
        assert hashlib.sha256(p.read_bytes()).hexdigest() == m["source_parquet_sha256"]
        data = pq.read_table(p).to_pydict()
        actions = native_actions(data["action"])
        for row in m["samples"]:
            step = row["step"]
            n = min(50, end - step)
            assert row["skill"] == "grasp" and step % 16 == 0 and 0 <= step < end
            with np.load(path.parent / row["actions"], allow_pickle=False) as target:
                assert target["actions"].shape == (50, 12) and target["valid"].dtype == np.bool_
                assert np.array_equal(target["valid"], np.arange(50) < n)
                assert np.array_equal(target["actions"][:n], actions[step : step + n])
                assert not target["actions"][n:].any()
                max_deadband = max(
                    max_deadband, float(np.abs(target["actions"][:n, 10]).max(initial=0))
                )
            expected = native_state(data["observation.state"][step])
            with np.load(path.parent / row["observation"], allow_pickle=False) as obs:
                for k, v in expected.items():
                    assert np.array_equal(obs[k], v)
                for camera in [
                    "robot0_agentview_left",
                    "robot0_agentview_right",
                    "robot0_eye_in_hand",
                ]:
                    image = obs["video." + camera]
                    assert image.dtype == np.uint8 and image.shape == (256, 256, 3)
            samples += 1
        datasets[m["source_dataset"]] = datasets.get(m["source_dataset"], 0) + 1
        episodes += 1
    assert episodes == report["episodes"] and samples == report["samples"] and max_deadband < 0.02
    return dict(
        passed=True,
        episodes=episodes,
        samples=samples,
        datasets=datasets,
        all_action_state_mask_alignment_verified=True,
        zero_std_dim10_max_abs=max_deadband,
        heldout_seed_collision=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = validate(Path(args.root), Path(args.source))
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
