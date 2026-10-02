"""Convert validated official human grasp prefixes into native pi05 BC shards."""

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np


def native_actions(actions):
    actions = np.asarray(actions, dtype=np.float32)
    assert actions.ndim == 2 and actions.shape[1] == 12 and np.isfinite(actions).all()
    # Official PandaOmron_modality.json -> native HDF5/OpenPI ordering.
    return actions[:, [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]]


def native_state(state):
    state = np.asarray(state, dtype=np.float32)
    assert state.shape == (16,) and np.isfinite(state).all()
    return {
        key: state[start:end].copy()
        for key, start, end in [
            ("state.base_position", 0, 3),
            ("state.base_rotation", 3, 7),
            ("state.end_effector_position_relative", 7, 10),
            ("state.end_effector_rotation_relative", 10, 14),
            ("state.gripper_qpos", 14, 16),
        ]
    }


def decode(video, end):
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-i",
        str(video),
        "-vf",
        f"trim=end_frame={end},select='not(mod(n,16))'",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    raw = subprocess.check_output(command)
    expected = (end + 15) // 16
    assert len(raw) == expected * 256 * 256 * 3, (video, len(raw), expected)
    return np.frombuffer(raw, np.uint8).reshape(expected, 256, 256, 3)


def build(source, validation, destination, allow_partial=False):
    import pyarrow.parquet as pq

    assert (
        (validation["complete"] or allow_partial)
        and not validation["gpu"]
        and validation["simulator_steps"] == 0
    )
    names = sorted(p.parent.name for p in source.glob("*/lerobot"))
    accepted = []
    destination.mkdir(parents=True, exist_ok=True)

    def build_one(row):
        dataset = source / row["dataset"] / "lerobot"
        ep = row["episode"]
        end = row["end"]
        episode_id = f"expert-human-{row['dataset']}-episode{ep:06d}"
        folder = destination / episode_id / "bc"
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "manifest.json").exists():
            previous = json.loads((folder / "manifest.json").read_text())
            assert (
                previous["source_parquet_sha256"] == row["parquet_sha256"]
                and previous["segments"][0]["end"] == end
            )
            return previous
        parquet = dataset / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet"
        assert hashlib.sha256(parquet.read_bytes()).hexdigest() == row["parquet_sha256"]
        table = pq.read_table(parquet).to_pydict()
        actions = native_actions(table["action"])
        assert len(actions) == row["length"]
        cameras = {
            key: decode(
                dataset
                / f"videos/chunk-{ep // 1000:03d}/observation.images.{key}/episode_{ep:06d}.mp4",
                end,
            )
            for key in ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
        }
        samples = []
        for index, step in enumerate(range(0, end, 16)):
            obs = native_state(table["observation.state"][step])
            obs.update({"video." + key: frames[index] for key, frames in cameras.items()})
            observation = f"obs-{step:06d}.npz"
            action = f"actions-{step:06d}.npz"
            np.savez_compressed(folder / observation, **obs)
            n = min(50, end - step)
            chunk = np.zeros((50, 12), np.float32)
            chunk[:n] = actions[step : step + n]
            np.savez_compressed(folder / action, actions=chunk, valid=np.arange(50) < n)
            samples.append(dict(skill="grasp", step=step, observation=observation, actions=action))
        manifest = dict(
            model="expert-human",
            seed=10000000 + names.index(row["dataset"]) * 100000 + ep,
            prompt=row["prompt"],
            dataset_role="expert_grasp_base",
            episode_id=episode_id,
            samples=samples,
            segments=[dict(skill="grasp", start=0, end=end)],
            source_dataset=row["dataset"],
            source_episode_index=ep,
            source_parquet_sha256=row["parquet_sha256"],
            native_grasp_validation=row,
            label_source="successful_public_human_expert_actions",
            action_mapping="PandaOmron LeRobot->native HDF5: [5..11,0..4]",
            state_mapping="named PandaOmron modality fields, XYZW quaternions unchanged",
            source_fps=20,
            chunk_stride=16,
            placement_actions_included=False,
        )
        temp = folder / "manifest.tmp"
        temp.write_text(json.dumps(manifest, indent=2))
        temp.replace(folder / "manifest.json")
        return manifest

    with ThreadPoolExecutor(max_workers=4) as pool:
        for manifest in pool.map(build_one, validation["accepted"]):
            accepted.append(manifest)
            if len(accepted) % 25 == 0:
                print(json.dumps({"built_episodes": len(accepted)}), flush=True)
    report = dict(
        complete=validation["complete"],
        episodes=len(accepted),
        samples=sum(len(m["samples"]) for m in accepted),
        skill="grasp",
        validation_rejected=len(validation["rejected"]),
        dataset_role="expert_grasp_base",
        online_success_target_contribution=0,
        expert_episode_ids=[m["episode_id"] for m in accepted],
    )
    (destination / "dataset.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-partial-validation", action="store_true")
    args = ap.parse_args()
    print(
        json.dumps(
            build(
                Path(args.source),
                json.loads(Path(args.validation).read_text()),
                Path(args.out),
                allow_partial=args.allow_partial_validation,
            )
        )
    )


if __name__ == "__main__":
    main()
