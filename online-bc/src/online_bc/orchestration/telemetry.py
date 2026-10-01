"""One central W&B writer for learner, collection, evaluation and timings."""

import argparse
import hashlib
import json
import netrc
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

from online_bc.data.data_control import atomic_json


def events(root, learner, steps=50):
    for row in learner.get("updates", []):
        version = int(row["round"].split("-")[-1])
        step = (version - 1) * steps + row["step"]
        yield (
            f"update/{version}/{row['step']}",
            {
                "train/global_step": step,
                "train/loss": row["loss"],
                "train/grad_norm": row["grad_norm"],
                "train/update_seconds": row["seconds"],
                "train/control_revision": row["control_revision"],
                "train/source_model": row["source_model"],
                "train/batch_size": row.get("batch_size", 1),
                "train/learning_rate": row.get("learning_rate", 1e-4),
            },
        )
    for path in sorted(root.glob("evaluation/version-*/done.json")):
        row = json.loads(path.read_text())
        yield (
            f"eval/{row['policy_version']}",
            {
                "eval/version": row["policy_version"],
                "eval/attempts": row["attempts"],
                "eval/cup_success_rate": row["cup_success_rate"],
                "eval/cup_given_grasp": row.get("cup_given_grasp"),
                "eval/grasp_success_rate": row["grasp_successes"] / row["attempts"],
                "eval/wall_seconds": row.get("wall_seconds")
                if row.get("wall_seconds") is not None
                else float("nan"),
            },
        )
    for path in sorted(root.glob("timing-*.json")):
        row = json.loads(path.read_text())
        yield path.stem, {f"pipeline/{key}": value for key, value in row.items()}
    for path in sorted(root.glob("collection-round-*.json")):
        row = json.loads(path.read_text())
        version = int(path.stem.split("-")[-1])
        for source in row["sources"]:
            yield (
                f"collection/{version}/{source['batch']}/{source['node']}",
                {
                    "collection/version": version,
                    "collection/node": source["node"],
                    "collection/attempts": len(source["seeds"]),
                    "collection/valid_successes": len(source["accepted"]),
                    **{
                        f"collection/{key}": value
                        for key, value in source.get("timings", {}).items()
                    },
                },
            )
    for path in sorted(root.glob("adaptation-version-*.json")):
        row = json.loads(path.read_text())
        yield path.stem, {f"adaptation/{key}": value for key, value in row.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--review-root", required=True)
    args = ap.parse_args()
    c = json.loads(Path(args.config).read_text())
    root = Path(c["root"])
    folder = root / "telemetry"
    folder.mkdir(parents=True, exist_ok=True)
    cursor = folder / "cursor.json"
    seen = set(json.loads(cursor.read_text())) if cursor.exists() else set()
    import wandb

    host = urlsplit(os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")).netloc
    credentials = netrc.netrc().authenticators(host)
    if credentials:
        os.environ["WANDB_API_KEY"] = credentials[2]

    run = wandb.init(
        project="coffee-online-bc",
        entity=c.get("wandb_entity", "aiden-lab-desktop"),
        id=hashlib.sha256(c["run"].encode()).hexdigest()[:12],
        name=c["run"],
        resume="allow",
        dir=str(folder),
        config=c,
        settings=wandb.Settings(init_timeout=60),
    )
    run.define_metric("train/*", step_metric="train/global_step")
    run.define_metric("eval/*", step_metric="eval/version")
    run.define_metric("collection/*", step_metric="collection/version")
    atomic_json(folder / "status.json", dict(url=run.url, status="online", at=time.time()))
    while True:
        try:
            path = Path(args.review_root) / "learner-state/state.json"
            learner = json.loads(path.read_text()) if path.exists() else {}
            for key, metrics in events(root, learner, c["steps_per_round"]):
                if key not in seen:
                    run.log({k: v for k, v in metrics.items() if v is not None})
                    seen.add(key)
                    atomic_json(cursor, sorted(seen))
            if (root / "status.json").exists():
                state = json.loads((root / "status.json").read_text())
                run.summary["pipeline_state"] = state
            run.summary["learner_state"] = learner.get("status", "waiting")
            health_path = root / "health.json"
            if health_path.exists():
                health = json.loads(health_path.read_text())
                run.summary["health"] = health
                key = f"health/{health['at']}"
                if key not in seen:
                    metrics = {
                        f"health/{name}_reachable": int(row["ok"])
                        for name, row in health["hosts"].items()
                    }
                    for name, row in health["hosts"].items():
                        values = row.get("gpu", "").split(",")
                        if values and values[0].strip().isdigit():
                            metrics[f"system/{name}_gpu_memory_mib"] = int(values[0])
                        if len(values) > 1 and values[1].strip().isdigit():
                            metrics[f"system/{name}_gpu_utilization"] = int(values[1])
                    run.log(metrics)
                    seen.add(key)
                    atomic_json(cursor, sorted(seen))
            atomic_json(
                folder / "status.json",
                dict(url=run.url, status="online", at=time.time(), logged_events=len(seen)),
            )
        except (OSError, ValueError, KeyError) as exc:
            print(json.dumps(dict(event="telemetry_retry", error=str(exc))), flush=True)
        time.sleep(10)


if __name__ == "__main__":
    main()
