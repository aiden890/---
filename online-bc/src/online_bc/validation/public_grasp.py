"""Validate public grasp cuts from recorded MuJoCo states, without rollouts or GPUs."""

import argparse
import fcntl
import gzip
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np


def validate_episode(source, row, collision_only=False):
    import mujoco

    if not row["full_success"]:
        raise ValueError("Full expert demonstration did not reach recorded task success")
    episode = source / row["dataset"] / "lerobot/extras" / f"episode_{row['episode']:06d}"
    states = np.load(episode / "states.npz", allow_pickle=False)["states"]
    xml = ET.fromstring(gzip.decompress((episode / "model.xml.gz").read_bytes()))
    for element in xml.iter():
        file = element.get("file")
        if file:
            for package in ["robosuite", "robocasa"]:
                marker = package + "/models/assets/"
                if marker in file:
                    element.set("file", "/opt/" + package + "/" + marker + file.split(marker, 1)[1])
    if collision_only:
        # Only group-1, non-colliding visual geoms and their unused assets.
        # Compiler inertia uses group 0; joint/state order is untouched.
        for parent in xml.iter():
            for element in list(parent):
                if (
                    element.tag == "geom"
                    and element.get("group") == "1"
                    and element.get("contype") == "0"
                    and element.get("conaffinity") == "0"
                ):
                    parent.remove(element)
        needed = {g.get("mesh") for g in xml.iter("geom") if g.get("mesh")}
        for asset in xml.findall("asset"):
            for element in list(asset):
                if element.tag in ["texture", "material"] or (
                    element.tag == "mesh" and element.get("name") not in needed
                ):
                    asset.remove(element)
        for geom in xml.iter("geom"):
            geom.attrib.pop("material", None)
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    data = mujoco.MjData(model)
    assert states.shape == (row["length"], 1 + model.nq + model.nv + model.na)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj_main")
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obj_joint0")
    assert body >= 0 and joint >= 0
    offset = int(model.jnt_qposadr[joint]) + 1
    rest = states[0, offset : offset + 3]
    object_geoms = set(np.flatnonzero(model.geom_bodyid == body).tolist())
    pads = [
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"gripper0_right_finger{i}_pad_collision"
        )
        for i in [1, 2]
    ]
    assert all(p >= 0 for p in pads)

    def predicate(index):
        state = states[index]
        data.time = float(state[0])
        data.qpos[:] = state[1 : 1 + model.nq]
        data.qvel[:] = state[1 + model.nq : 1 + model.nq + model.nv]
        if model.na:
            data.act[:] = state[-model.na :]
        mujoco.mj_forward(model, data)
        touched = set()
        for contact in data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if a in object_geoms:
                touched.add(b)
            if b in object_geoms:
                touched.add(a)
        motion = float(np.linalg.norm(data.xpos[body] - rest))
        return dict(
            bilateral_grasp=all(p in touched for p in pads),
            mug_motion=motion,
            mug_world_position=data.xpos[body].tolist(),
        )

    end = row["annotated_end"]
    if end is None:
        # Atomic datasets have no subtask annotations. Require the same
        # bilateral fingerpad contact and >10cm motion used by native collection.
        candidates = np.flatnonzero(
            np.linalg.norm(states[:, offset : offset + 3] - rest, axis=1) > 0.10
        )
        for index in candidates:
            check = predicate(int(index))
            if check["bilateral_grasp"] and check["mug_motion"] > 0.10:
                end = int(index)
                break
    if end is None or not 0 < end < len(states):
        raise ValueError("No validated grasp boundary")
    check = predicate(end)
    if not check["bilateral_grasp"] or check["mug_motion"] <= 0.10:
        raise ValueError("Annotated cut failed native bilateral-grasp and motion predicate")
    return dict(
        row,
        end=end,
        start=0,
        native_predicate=check,
        boundary_source="official_human_pick_annotation"
        if row["annotated_end"] is not None
        else "recorded_state_native_grasp_motion",
        actions_interval="[start,end); state at end proves grasp; placement actions excluded",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--collision-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "singleton.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "started.json").exists():
            raise RuntimeError("Already started; preserve existing report")
        (out / "started.json").write_text(
            json.dumps(
                {
                    "started_at": time.time(),
                    "gpu": False,
                    "gradient_updates": 0,
                    "simulator_steps": 0,
                }
            )
        )
        accepted, rejected = [], []
        rows = json.loads(Path(args.candidates).read_text())
        for index, row in enumerate(rows):
            try:
                accepted.append(
                    validate_episode(Path(args.source), row, collision_only=args.collision_only)
                )
            except Exception as error:
                rejected.append(dict(row, reason=str(error)))
            if index % 25 == 0 or index == len(rows) - 1:
                report = dict(
                    complete=index == len(rows) - 1,
                    attempted=index + 1,
                    accepted=accepted,
                    rejected=rejected,
                    gpu=False,
                    gradient_updates=0,
                    simulator_steps=0,
                )
                temp = out / "report.tmp"
                temp.write_text(json.dumps(report, indent=2))
                temp.replace(out / "report.json")
                print(
                    json.dumps(
                        {
                            "attempted": index + 1,
                            "accepted": len(accepted),
                            "rejected": len(rejected),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
