"""Rollout-side collection, direct shard upload, and between-batch reload."""

from online_bc.paths import PROJECT_ROOT

import argparse
import json
import pickle
import subprocess
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def profile_summary(results):
    profiles = [row["phase_seconds"] for row in results if row.get("phase_seconds")]
    summary = {"profiled_episodes": len(profiles)}
    for key in sorted({key for row in profiles for key in row}):
        values = [row[key] for row in profiles if key in row]
        summary[f"episode_mean_{key}_seconds"] = sum(values) / len(values)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("action", choices=["collect", "reload", "eval"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=30)
    args = ap.parse_args()
    began = time.monotonic()
    c = json.loads(Path(args.config).read_text())
    model = c["model"]
    fields = dict(
        round=args.round, batch=args.batch, model=model, run=args.run, url=c["policy_url"]
    )
    root = Path(c["output_root"]) / f"round-{args.round:04d}" / f"batch-{args.batch:02d}"
    if args.action == "eval":
        root = Path(c["output_root"]).parent / "evaluation" / f"version-{args.round:04d}"
    root.mkdir(parents=True, exist_ok=True)

    def render(argv, **extra):
        return [x.format(**fields, **extra) for x in argv]

    def sync(direction, folder, prefix):
        cmd = render(c["transport_argv"], direction=direction, directory=str(folder), prefix=prefix)
        subprocess.run(cmd, check=True)

    if args.dry_run:
        print(json.dumps(dict(model=model, action=args.action, round=args.round, root=str(root))))
        return
    if args.action == "reload":
        dest = Path(c["adapter_root"]) / f"round-{args.round:04d}"
        sync("download", dest, f"{args.run}/weights/{model}/round-{args.round:04d}")
        subprocess.run(render(c["ensure_policy_argv"]), check=True)
        for _ in range(120):
            try:
                with urllib.request.urlopen(c["policy_url"], timeout=2) as response:
                    json.load(response)
                break
            except OSError:
                time.sleep(2)
        else:
            raise RuntimeError("Policy server did not restart for reload")
        req = urllib.request.Request(
            c["policy_url"],
            data=pickle.dumps(
                dict(op="load", path=c["adapter_container_root"] + f"/round-{args.round:04d}"),
                protocol=4,
            ),
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            result = pickle.loads(r.read())
        assert result["version"] == args.round
        (PROJECT_ROOT / "current-adapter.json").write_text(
            json.dumps(
                dict(
                    version=args.round,
                    container_path=c["adapter_container_root"] + f"/round-{args.round:04d}",
                )
            )
        )
        print(json.dumps(dict(model=model, reloaded=args.round)))
        return
    subprocess.run(render(c["ensure_policy_argv"]), check=True)
    for _ in range(120):
        try:
            with urllib.request.urlopen(c["policy_url"], timeout=2) as r:
                health = json.load(r)
            assert health["model"] == model and health["version"] == (
                args.round if args.action == "eval" else args.round - 1
            )
            break
        except OSError:
            time.sleep(2)
    else:
        raise RuntimeError("Policy server did not become ready")
    seeds = [
        993000 + args.round * 100 + args.seed_offset + i
        for i in range(1, (args.episodes or c.get("episodes_per_round", 8)) + 1)
    ]
    if args.action == "eval":
        assert 1 <= args.eval_episodes <= 30
        seeds = list(range(992001, 992001 + args.eval_episodes))
    shards = [seeds[i :: c.get("workers", 3)] for i in range(c.get("workers", 3))]

    def collect(pair):
        index, seeds = pair
        if not seeds:
            return
        with (root / f"worker-{index}.log").open("w") as log:
            command = render(c["collector_argv"], seeds=",".join(map(str, seeds)))
            if args.action == "eval":
                command[command.index("--out") + 1] = (
                    f"/results/coffee-online-bc/evaluation/version-{args.round:04d}/{model}"
                )
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    with ThreadPoolExecutor(max_workers=c.get("workers", 3)) as pool:
        list(pool.map(collect, enumerate(shards)))
    collection_seconds = time.monotonic() - began
    results = [
        json.loads(path.read_text()) for path in sorted((root / model).glob("*/result.json"))
    ]
    profile = profile_summary(results)
    profile["simulator_workers"] = sum(bool(shard) for shard in shards)
    if args.action == "eval":
        assert len(results) == len(seeds) and {row["seed"] for row in results} == set(seeds)
        assert all(row["policy_version"] == args.round for row in results)
        successes = sum(row["cup_placed"] for row in results)
        grasped = sum("mug_grasped" in row["milestones"] for row in results)
        report = dict(
            model=model,
            policy_version=args.round,
            attempts=len(seeds),
            cup_successes=successes,
            cup_success_rate=successes / len(seeds),
            grasp_successes=grasped,
            cup_given_grasp=successes / grasped if grasped else None,
            seeds=seeds,
            training_data=False,
            profiling=profile,
            wall_seconds=collection_seconds,
            outcomes=[
                dict(
                    seed=row["seed"],
                    cup_placed=row["cup_placed"],
                    grasped="mug_grasped" in row["milestones"],
                )
                for row in results
            ],
        )
        evaluation_upload = root / "evaluation-upload"
        evaluation_upload.mkdir(exist_ok=True)
        (evaluation_upload / "metrics.json").write_text(json.dumps(report, indent=2))
        sync("upload", evaluation_upload, f"{args.run}/evaluation/{model}/version-{args.round:04d}")
        print(json.dumps(report), flush=True)
        return
    upload = root / "upload"
    shard_root = root / model
    if c.get("skill") == "cup_placement":
        stage = time.monotonic()
        subprocess.run(render(c["build_dataset_argv"]), check=True)
        dataset_seconds = time.monotonic() - stage
        shard_root = root / "cup-dataset" / model
    else:
        dataset_seconds = 0
    stage = time.monotonic()
    subprocess.run(
        [
            c.get("host_python", "python3"),
            "-m",
            "online_bc.data.pack_shards",
            str(shard_root),
            "--out",
            str(upload),
        ],
        check=True,
    )
    packing_seconds = time.monotonic() - stage
    prefix = f"{args.run}/data/{model}/{c.get('node', model)}/round-{args.round:04d}/batch-{args.batch:02d}"
    stage = time.monotonic()
    sync("upload", upload, prefix)
    upload_seconds = time.monotonic() - stage
    dataset = json.loads((root / "cup-dataset" / model / "dataset.json").read_text())
    print(
        json.dumps(
            dict(
                model=model,
                node=c.get("node", model),
                batch=args.batch,
                round=args.round,
                prefix=prefix,
                seeds=seeds,
                accepted=[x["episode"] for x in dataset["accepted"]],
                remote_root=str(root),
                timings=dict(
                    collection_seconds=collection_seconds,
                    dataset_seconds=dataset_seconds,
                    packing_seconds=packing_seconds,
                    upload_seconds=upload_seconds,
                    total_seconds=time.monotonic() - began,
                    **profile,
                ),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
