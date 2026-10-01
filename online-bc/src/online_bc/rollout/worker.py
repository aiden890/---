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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("action", choices=["collect", "reload"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    c = json.loads(Path(args.config).read_text())
    model = c["model"]
    fields = dict(round=args.round, model=model, run=args.run, url=c["policy_url"])
    root = Path(c["output_root"]) / f"round-{args.round:04d}"
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
            assert health["model"] == model and health["version"] == args.round - 1
            break
        except OSError:
            time.sleep(2)
    else:
        raise RuntimeError("Policy server did not become ready")
    seeds = [993000 + args.round * 100 + i for i in range(1, c.get("episodes_per_round", 8) + 1)]
    shards = [seeds[i :: c.get("workers", 3)] for i in range(c.get("workers", 3))]

    def collect(pair):
        index, seeds = pair
        if not seeds:
            return
        with (root / f"worker-{index}.log").open("w") as log:
            subprocess.run(
                render(c["collector_argv"], seeds=",".join(map(str, seeds))),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    with ThreadPoolExecutor(max_workers=c.get("workers", 3)) as pool:
        list(pool.map(collect, enumerate(shards)))
    upload = root / "upload"
    shard_root = root / model
    if c.get("skill") == "cup_placement":
        subprocess.run(render(c["build_dataset_argv"]), check=True)
        shard_root = root / "cup-dataset" / model
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
    sync("upload", upload, f"{args.run}/data/{model}/round-{args.round:04d}")


if __name__ == "__main__":
    main()
