#!/usr/bin/env python3
"""Back up old completed rounds to HF before pruning local observation NPZ files."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time


def eligible(root, keep_rounds=2):
    current = json.loads((root / "current-adapter.json").read_text())["version"]
    for folder in sorted((root / "rollouts").glob("round-*")):
        if len(folder.name) != len("round-0000") or not folder.name[6:].isdigit():
            continue
        number = int(folder.name[6:])
        if number > current - keep_rounds:
            continue
        batches = list(folder.glob("batch-*"))
        if batches and all(
            (batch / "pi05/complete.json").exists()
            and (batch / "upload/manifest.json").exists()
            for batch in batches
        ):
            yield folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--archive-root", type=Path, required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--token-file", type=Path, required=True)
    ap.add_argument("--bucket", default="khmin101/vla-rollout-transfer")
    ap.add_argument("--cleanup-image", required=True)
    args = ap.parse_args()
    args.archive_root.mkdir(parents=True, exist_ok=True)
    lock = (args.archive_root / "archiver.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.nice(10)
    os.environ["HF_HOME"] = str(args.archive_root / "hf-cache")
    os.environ["HF_XET_CACHE"] = str(args.archive_root / "xet-cache")
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    from huggingface_hub import get_bucket_file_metadata, sync_bucket

    token = args.token_file.read_text().strip()
    assert token
    while True:
        for folder in eligible(args.root):
            stage = args.archive_root / "cloud-stage" / folder.name
            stage.mkdir(parents=True, exist_ok=True)
            done = stage / "archived.json"
            if done.exists():
                continue
            try:
                pending = stage / "uploaded.json"
                target = f"{args.prefix}/{folder.name}"
                if pending.exists():
                    report = json.loads(pending.read_text())
                else:
                    archive = stage / "rollouts.tar.gz"
                    with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
                        tar.add(folder.resolve(), arcname=folder.name)
                    digest = hashlib.sha256()
                    with archive.open("rb") as stream:
                        for block in iter(lambda: stream.read(1048576), b""):
                            digest.update(block)
                    report = dict(
                        round=int(folder.name[6:]), bytes=archive.stat().st_size,
                        sha256=digest.hexdigest(), bucket=args.bucket,
                        remote_path=target + "/rollouts.tar.gz", local_round=str(folder),
                    )
                    (stage / "SHA256.json").write_text(json.dumps(report, indent=2))
                    sync_bucket(str(stage), f"hf://buckets/{args.bucket}/{target}",
                                token=token, quiet=True)
                    remote = get_bucket_file_metadata(
                        args.bucket, report["remote_path"], token=token
                    )
                    assert remote.size == report["bytes"]
                    report["remote_size_verified"] = True
                    pending.write_text(json.dumps(report, indent=2))
                # Retry after a crash only after rechecking the remote object.
                remote = get_bucket_file_metadata(
                    args.bucket, report["remote_path"], token=token
                )
                assert remote.size == report["bytes"]
                # Scope the privileged container to this one old, completed round.
                cleanup = (
                    "import pathlib,json; r=pathlib.Path('/old-round'); "
                    "files=list(r.rglob('*.npz')); "
                    "sizes=[p.stat().st_size for p in files]; "
                    "[p.unlink() for p in files]; "
                    "print(json.dumps({'removed_npz':len(files),'pruned_npz_bytes':sum(sizes)}))"
                )
                result = subprocess.run(
                    ["docker", "run", "--rm", "-v", str(folder.resolve()) + ":/old-round",
                     args.cleanup_image, "python3", "-c", cleanup],
                    capture_output=True, text=True, check=True,
                )
                report.update(json.loads(result.stdout.strip().splitlines()[-1]))
                report["archived_at"] = time.time()
                report["videos_actions_traces_manifests_retained"] = True
                done.write_text(json.dumps(report, indent=2))
                (folder / "observation-archive.json").write_text(json.dumps(report, indent=2))
                (stage / "rollouts.tar.gz").unlink(missing_ok=True)
                print(json.dumps(dict(event="round_observations_archived", **report)), flush=True)
            except Exception as error:
                print(json.dumps(dict(event="archive_retry", round=folder.name,
                                      error_type=type(error).__name__)), flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
