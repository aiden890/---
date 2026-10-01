"""HF-backed jobs: collect → native BC → adapter publication. One service/model."""

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import time
import fcntl
import os
from pathlib import Path
from online_bc.transport.control_sync import ControlSync
from online_bc.data.data_control import atomic_json


def unpack(folder, destination):
    archive = folder / "bc-shards.tar.gz"
    m = json.loads((folder / "manifest.json").read_text())
    h = hashlib.sha256()
    with archive.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    assert h.hexdigest() == m["sha256"], "Corrupt data transfer"
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            p = Path(member.name)
            if p.is_absolute() or ".." in p.parts or not member.isfile():
                raise ValueError(f"Unsafe shard member: {member.name}")
        tar.extractall(destination, filter="data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--transport-python", required=True)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--run", default="coffee-online-bc-20261001")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--poll", type=int, default=10)
    ap.add_argument("--gpu-slot", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    ap.add_argument(
        "--gpu-lock-root", default=str(Path.home() / ".cache/coffee-online-bc-gpu-locks")
    )
    ap.add_argument("--bootstrap-models", nargs="+", default=["xiaomi", "pi05", "groot"])
    ap.add_argument("--bootstrap-steps", type=int, default=50)
    ap.add_argument("--skills", nargs="+", default=["cup_placement"])
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    def sync(direction, folder, prefix):
        return subprocess.run(
            [
                args.transport_python,
                "-m",
                "online_bc.transport.hf_transfer",
                direction,
                str(folder),
                prefix,
                "--token-file",
                args.token_file,
            ],
            capture_output=True,
            text=True,
        )

    control = ControlSync(root, args.run, sync)
    control.start()
    for model in args.bootstrap_models:
        folder = root / "incoming" / model / "bootstrap"
        r = sync("download", folder, f"{args.run}/bootstrap/{model}")
        if r.returncode:
            raise RuntimeError(f"Bootstrap data unavailable for {model}: {r.stderr[-500:]}")
        unpack(folder, root / "data" / model / "bootstrap")
    status = root / "service-status.json"
    saved = json.loads(status.read_text()) if status.exists() else {}
    resume = saved.get("checkpoint")
    start = int(saved.get("completed_round", 0)) + 1
    if resume is None and args.bootstrap_steps:
        roots = [str(p) for p in (root / "data").glob("*/bootstrap")]
        out = root / "training" / "bootstrap"
        cmd = [
            sys.executable,
            "-m",
            "online_bc.learning.train_online_bc",
            "--model",
            args.model,
            "--checkpoint",
            args.checkpoint,
            "--data",
            *roots,
            "--out",
            str(out),
            "--steps",
            str(args.bootstrap_steps),
            "--save-every",
            str(args.bootstrap_steps),
            "--skills",
            *args.skills,
        ]
        locks = Path(args.gpu_lock_root)
        locks.mkdir(parents=True, exist_ok=True)
        with (locks / f"gpu-{hashlib.sha256(args.gpu_slot.encode()).hexdigest()[:16]}.lock").open(
            "a"
        ) as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = args.gpu_slot
            subprocess.run(cmd, check=True, env=env)
        resume = (out / "latest").read_text().strip()
        adapter = Path(resume)
        meta = json.loads((adapter / "metadata.json").read_text())
        meta.update(step=0, phase="bootstrap", optimizer_steps_this_round=args.bootstrap_steps)
        (adapter / "metadata.json").write_text(json.dumps(meta, indent=2))
        (adapter / "updates.json").write_bytes((out / "updates.json").read_bytes())
        result = sync("upload", adapter, f"{args.run}/weights/{args.model}/round-0000")
        if result.returncode:
            raise RuntimeError(result.stderr[-1000:])
        (root / "service-status.json").write_text(
            json.dumps(dict(model=args.model, completed_round=0, checkpoint=resume), indent=2)
        )
    if resume is None and not args.bootstrap_steps:
        atomic_json(
            status,
            dict(
                model=args.model,
                completed_round=0,
                checkpoint=None,
                bootstrap_skipped=True,
                status="waiting_for_rollouts",
            ),
        )
    for index in range(start, args.rounds + 1):
        job_dir = root / f"jobs/round-{index:04d}"
        job = None
        while job is None:
            r = sync("download", job_dir, f"{args.run}/jobs/{args.model}/round-{index:04d}")
            if r.returncode == 0 and (job_dir / "job.json").exists():
                job = json.loads((job_dir / "job.json").read_text())
            else:
                time.sleep(args.poll)
        assert job["model"] == args.model and job["round"] == index
        roots = []
        for source in job["data_prefixes"]:
            source_key = f"round-{source['round']:04d}-{source.get('node', source['model'])}-batch-{source.get('batch', 1):02d}"
            folder = root / "incoming" / source["model"] / source_key
            r = sync("download", folder, source["prefix"])
            if r.returncode:
                raise RuntimeError(
                    f"Data download failed for {source['prefix']}: {r.stderr[-1000:]}"
                )
            data = root / "data" / source["model"] / source_key
            unpack(folder, data)
            roots.append(str(data))
        # Include past rounds: Replay applies a bounded episode window.
        roots = [str(p) for p in (root / "data").glob("*/*")]
        atomic_json(
            status,
            dict(
                model=args.model,
                completed_round=index - 1,
                checkpoint=resume,
                active_round=index,
                status="training",
            ),
        )
        out = root / "training" / f"round-{index:04d}"
        cmd = [
            sys.executable,
            "-m",
            "online_bc.learning.train_online_bc",
            "--model",
            args.model,
            "--checkpoint",
            args.checkpoint,
            "--data",
            *roots,
            "--out",
            str(out),
            "--steps",
            str(job.get("steps", 50)),
            "--save-every",
            str(job.get("steps", 50)),
        ]
        cmd += ["--lr", str(job.get("lr", 1e-4))]
        cmd += [
            "--batch-size",
            str(job.get("batch_size", int(os.environ.get("PI_CUP_BATCH_SIZE", "1")))),
        ]
        cmd += [
            "--skills",
            *job.get("skills", ["cup_placement", "button_press"]),
            "--controls",
            str(control.path),
            "--control-health",
            str(control.health),
        ]
        if resume:
            training_resume = resume
        else:
            training_resume = None
        if job.get("resume_round") is not None:
            best = int(job["resume_round"])
            training_resume = (
                (root / "training" / f"round-{best:04d}" / "latest").read_text().strip()
                if best
                else None
            )
        if training_resume:
            cmd += ["--resume", training_resume]
        locks = Path(args.gpu_lock_root)
        locks.mkdir(parents=True, exist_ok=True)
        with (locks / f"gpu-{hashlib.sha256(args.gpu_slot.encode()).hexdigest()[:16]}.lock").open(
            "a"
        ) as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = args.gpu_slot
            subprocess.run(cmd, check=True, env=env)
        resume = (out / "latest").read_text().strip()
        adapter = Path(resume)
        # Global deployment version must advance across independent learner jobs.
        meta = json.loads((adapter / "metadata.json").read_text())
        meta["step"] = index
        meta["optimizer_steps_this_round"] = job.get("steps", 50)
        meta["resumed_from_round"] = job.get("resume_round")
        (adapter / "updates.json").write_bytes((out / "updates.json").read_bytes())
        (adapter / "metadata.json").write_text(json.dumps(meta, indent=2))
        r = sync("upload", adapter, f"{args.run}/weights/{args.model}/round-{index:04d}")
        if r.returncode:
            raise RuntimeError(r.stderr[-1000:])
        atomic_json(
            status,
            dict(
                model=args.model,
                completed_round=index,
                checkpoint=resume,
                status="waiting_for_rollouts",
            ),
        )
        control.once()


if __name__ == "__main__":
    main()
