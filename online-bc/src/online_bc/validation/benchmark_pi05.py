"""Compare native batch throughput with disposable adapters under the GPU lock."""

import argparse
import fcntl
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from online_bc.data.data_control import atomic_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4])
    args = ap.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    lock_root = Path.home() / ".cache/coffee-online-bc-gpu-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    rows = []
    with (lock_root / f"gpu-{hashlib.sha256(gpu.encode()).hexdigest()[:16]}.lock").open(
        "a"
    ) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for batch in args.batches:
            out = root / f"batch-{batch}"
            out.mkdir(exist_ok=True)
            began = time.monotonic()
            with (out / "process.log").open("w") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "online_bc.learning.train_online_bc",
                        "--model",
                        "pi05",
                        "--checkpoint",
                        args.checkpoint,
                        "--data",
                        *args.data,
                        "--out",
                        str(out),
                        "--steps",
                        "6",
                        "--save-every",
                        "6",
                        "--batch-size",
                        str(batch),
                        "--skills",
                        "cup_placement",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            row = dict(
                batch_size=batch,
                passed=result.returncode == 0,
                wall_seconds=time.monotonic() - began,
            )
            if result.returncode == 0:
                updates = json.loads((out / "updates.json").read_text())
                warm = [item["seconds"] for item in updates[2:]]
                row.update(
                    warm_median_seconds=statistics.median(warm),
                    samples_per_second=batch / statistics.median(warm),
                    losses=[item["loss"] for item in updates],
                )
            else:
                row["error_tail"] = (out / "process.log").read_text()[-2000:]
            rows.append(row)
            passed = [item for item in rows if item["passed"]]
            atomic_json(
                root / "report.json",
                dict(
                    rows=rows,
                    complete=False,
                    best_batch_size=max(passed, key=lambda x: x["samples_per_second"])["batch_size"]
                    if passed
                    else None,
                    deployed=False,
                ),
            )
            print(json.dumps(row), flush=True)
    report = json.loads((root / "report.json").read_text())
    report["complete"] = True
    atomic_json(root / "report.json", report)


if __name__ == "__main__":
    main()
