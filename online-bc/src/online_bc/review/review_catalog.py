"""Small metadata catalog; observations/actions stay on rollout hosts and HF."""

import argparse
import json
import subprocess
from pathlib import Path
from online_bc.data.data_control import atomic_json


def scan(root):
    rows = []
    for p in sorted(Path(root).glob("*/*/bc/manifest.json")):
        d = json.loads(p.read_text())
        segment = d["segments"][0]
        rows.append(
            dict(
                id=f"{d['model']}-seed{d['seed']}",
                model=d["model"],
                seed=d["seed"],
                samples=len(d["samples"]),
                start=segment["start"],
                end=segment["end"],
                eligible=True,
                origin="initial_success_data",
                round=0,
                policy_version=d.get("source_policy_version", 0),
            )
        )
    return rows


def initial(tracking, rows):
    t = Path(tracking)
    metrics = json.loads(
        (t / "media/preparecoffee-three-models-20261001/skill-metrics.json").read_text()
    )
    clips = json.loads((t / "media/pi05-cup-bc-20261001/manifest.json").read_text())["examples"]
    clipmap = {x["episode"]: x["url"] for x in clips}
    accepted = {x["id"]: x for x in rows}
    catalog = []
    for ep in sorted(metrics["episodes"], key=lambda x: (x["model"], x["seed"])):
        eid = f"{ep['model']}-seed{ep['seed']}"
        row = dict(
            id=eid,
            model=ep["model"],
            seed=ep["seed"],
            samples=0,
            eligible=False,
            origin="baseline_evaluation_only",
            round=0,
            policy_version=0,
            cup_success=ep["cup_placed"],
            video=f"http://100.86.183.64:8899/media/preparecoffee-three-models-20261001/{eid}/video.mp4",
            failure_stage=ep.get("failure_stage"),
        )
        row.update(accepted.get(eid, {}))
        if eid in clipmap:
            row["clip"] = "http://100.86.183.64:8899" + clipmap[eid]
        catalog.append(row)
    atomic_json(t / "reports/pi-cup-data-catalog.json", {"episodes": catalog})


def publish_round(config, round_index, source=None):
    c = json.loads(Path(config).read_text())
    t = Path(c["tracking_root"])
    worker = c["workers"][source["node"]] if source else None
    w = json.loads(
        Path(worker["worker_config"] if worker else c["review_worker_config"]).read_text()
    )
    host = worker["host"] if worker else c["review_worker_host"]
    remote = (
        Path(source["remote_root"])
        if source
        else Path(w["output_root"]) / f"round-{round_index:04d}"
    )
    dest = t / "media/pi-cup-online-rollouts" / f"round-{round_index:04d}"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "scp",
            f"{host}:{remote}/cup-dataset/pi05/dataset.json",
            str(dest / f"dataset-{source['node']}-{source['batch']:02d}.json")
            if source
            else str(dest / "dataset.json"),
        ],
        check=True,
    )
    accepted = {
        x["episode"]: x
        for x in json.loads(
            (
                dest
                / (
                    f"dataset-{source['node']}-{source['batch']:02d}.json"
                    if source
                    else "dataset.json"
                )
            ).read_text()
        )["accepted"]
    }
    rows = []
    for seed in (
        source["seeds"]
        if source
        else [993000 + 100 * round_index + i for i in range(1, w["episodes_per_round"] + 1)]
    ):
        eid = f"pi05-seed{seed}"
        folder = dest / eid
        folder.mkdir(exist_ok=True)
        for name in ["video.mp4", "result.json"]:
            subprocess.run(
                ["scp", f"{host}:{remote}/pi05/{eid}/{name}", str(folder / name)], check=True
            )
        result = json.loads((folder / "result.json").read_text())
        a = accepted.get(eid)
        rows.append(
            dict(
                id=eid,
                model="pi05",
                seed=seed,
                round=round_index,
                policy_version=result.get("policy_version", round_index - 1),
                eligible=a is not None,
                samples=a["samples"] if a else 0,
                start=a["start"] if a else None,
                end=a["end"] if a else None,
                origin="online_rollout",
                cup_success=a is not None,
                video=f"http://100.86.183.64:8899/media/pi-cup-online-rollouts/round-{round_index:04d}/{eid}/video.mp4",
            )
        )
    catalog = t / "reports/pi-cup-data-catalog.json"
    d = json.loads(catalog.read_text())
    replace = {x["id"] for x in rows}
    d["episodes"] = [x for x in d["episodes"] if x["id"] not in replace] + rows
    atomic_json(catalog, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scan")
    p.add_argument("--tracking")
    p.add_argument("--initial")
    p.add_argument("--config")
    p.add_argument("--round", type=int)
    p.add_argument("--source-json")
    a = p.parse_args()
    if a.scan:
        print(json.dumps(scan(a.scan)))
    elif a.initial:
        initial(a.tracking, json.loads(Path(a.initial).read_text()))
    else:
        publish_round(a.config, a.round, json.loads(a.source_json) if a.source_json else None)
