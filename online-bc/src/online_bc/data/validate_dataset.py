"""Audit every camera/state/action chunk, including original action alignment."""

import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from online_bc.data.build_cup_dataset import INSTRUCTION


def validate(root, source_required=False, allow_empty=False):
    episodes = samples = aligned = 0
    max_deadband = 0.0
    seeds = []
    for path in sorted(Path(root).glob("*/bc/manifest.json")):
        m = json.loads(path.read_text())
        assert m["prompt"] == INSTRUCTION
        assert m["metrics"]["cup_placed"]
        assert len(m["segments"]) == 1
        seg = m["segments"][0]
        assert seg["skill"] == "cup_placement"
        assert seg["start"] % 16 == 0
        assert seg["start"] < seg["end"]
        source = Path(m["source_episode"])
        original = None
        if source.exists():
            original = np.load(source / "actions.npy", allow_pickle=False)
            trace = json.loads((source / "trace.json").read_text())
            assert (
                hashlib.sha256((source / "trace.json").read_bytes()).hexdigest()
                == m["source_trace_sha256"]
            )
            assert (
                trace[seg["start"] - 1]["mug_grasped"]
                and trace[seg["start"] - 1]["mug_motion"] > 0.1
            )
            assert not any(r["coffee_machine_on"] for r in trace[: seg["end"]])
            assert m["metrics"]["cup_placement_step"] == seg["end"]
        elif source_required:
            raise AssertionError("Original episode is unavailable")
        for row in m["samples"]:
            step = row["step"]
            assert (
                row["skill"] == "cup_placement"
                and step % 16 == 0
                and seg["start"] <= step < seg["end"]
            )
            with np.load(path.parent / row["actions"], allow_pickle=False) as f:
                a = f["actions"]
                v = f["valid"]
                n = min(50, seg["end"] - step)
                assert a.shape == (50, 12) and v.shape == (50,) and v.dtype == np.bool_
                assert (
                    np.array_equal(v, np.arange(50) < n)
                    and np.isfinite(a).all()
                    and not a[n:].any()
                )
                if original is not None:
                    assert np.array_equal(a[:n], original[step : step + n])
                    aligned += 1
                max_deadband = max(max_deadband, float(np.abs(a[:n, 10]).max(initial=0)))
            with np.load(path.parent / row["observation"], allow_pickle=False) as f:
                for cam in [
                    "robot0_agentview_left",
                    "robot0_agentview_right",
                    "robot0_eye_in_hand",
                ]:
                    im = f["video." + cam]
                    assert im.ndim == 3 and im.shape[-1] == 3 and im.dtype == np.uint8
                keys = [
                    "state.end_effector_position_relative",
                    "state.end_effector_rotation_relative",
                    "state.base_position",
                    "state.base_rotation",
                    "state.gripper_qpos",
                ]
                state = np.concatenate([f[k] for k in keys])
                assert state.shape == (16,) and np.isfinite(state).all()
                history = f["xiaomi/state_history"]
                assert history.shape == (4, 14) and np.isfinite(history).all()
                # Xiaomi uses axis-angle EE-first 14D, while native pi uses quaternion 16D.
                assert np.allclose(state[:3], history[-1, :3], atol=1e-6)
                assert np.allclose(state[-2:], history[-1, 6:8], atol=1e-6)
                assert np.allclose(state[7:10], history[-1, 8:11], atol=1e-6)
            samples += 1
        episodes += 1
        seeds.append(m["seed"])
    assert (samples > 0 or allow_empty) and max_deadband < 0.02
    return dict(
        passed=True,
        episodes=episodes,
        samples=samples,
        action_alignment_checked=aligned,
        zero_std_dim10_max_abs=max_deadband,
        seeds=seeds,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--source-required", action="store_true")
    p.add_argument("--allow-empty", action="store_true")
    p.add_argument("--out")
    a = p.parse_args()
    r = validate(a.root, a.source_required, a.allow_empty)
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=2))
    print(json.dumps(r))
