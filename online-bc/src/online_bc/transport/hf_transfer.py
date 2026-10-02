"""File transport in a separate huggingface_hub>=2 environment; never log tokens."""

import argparse
import hashlib
import json
from pathlib import Path
import os
import re

os.environ.setdefault("HF_HOME", str(Path.home() / ".cache/pi-cup-hf-transport"))
from huggingface_hub import sync_bucket  # noqa: E402 -- set cache location before importing Hub

p = argparse.ArgumentParser()
p.add_argument("direction", choices=["upload", "download"])
p.add_argument("directory")
p.add_argument("prefix")
p.add_argument("--bucket", default="khmin101/vla-rollout-transfer")
p.add_argument("--token-file", required=True)
p.add_argument("--inference-only", action="store_true")
a = p.parse_args()
assert a.prefix and all(x not in ("", ".", "..") for x in a.prefix.split("/"))
if a.inference_only and (
    a.direction != "download" or not re.search(r"/weights/pi05/round-[0-9]{4}$", a.prefix)
):
    p.error("--inference-only is restricted to pi05 checkpoint downloads")
directory = Path(a.directory)
directory.mkdir(parents=True, exist_ok=True)
remote = f"hf://buckets/{a.bucket}/{a.prefix}"
token = Path(a.token_file).read_text().strip()
assert token
source, destination = (
    (str(directory), remote) if a.direction == "upload" else (remote, str(directory))
)
options = {}
if a.inference_only:
    options["include"] = ["metadata.json", "pi05-lora.npz", "updates.json"]
sync_bucket(source, destination, token=token, quiet=True, **options)
if a.inference_only:
    assert all((directory / name).is_file() for name in options["include"])
files = {}
for f in sorted(directory.rglob("*")):
    if f.is_file():
        h = hashlib.sha256()
        with f.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        files[str(f.relative_to(directory))] = {"bytes": f.stat().st_size, "sha256": h.hexdigest()}
print(json.dumps({"direction": a.direction, "prefix": a.prefix, "files": files,
                  "inference_only": a.inference_only}))
