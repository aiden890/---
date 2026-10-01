"""Create one immutable, hash-verified upload per rollout server."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--out", required=True)
args = ap.parse_args()
root = Path(args.root)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
episodes = 0
samples = 0
with tarfile.open(out / "bc-shards.tar.gz", "w:gz") as tar:
    for path in sorted(root.glob("*/bc/manifest.json")):
        m = json.loads(path.read_text())
        if not m["samples"]:
            continue
        files = {"manifest.json"}
        for row in m["samples"]:
            files.update((row["observation"], row["actions"]))
        for name in sorted(files):
            tar.add(path.parent / name, arcname=path.parent.parent.name + "/bc/" + name)
        episodes += 1
        samples += len(m["samples"])
h = hashlib.sha256()
with (out / "bc-shards.tar.gz").open("rb") as f:
    for block in iter(lambda: f.read(1024 * 1024), b""):
        h.update(block)
(out / "manifest.json").write_text(
    json.dumps(
        dict(
            episodes=episodes,
            samples=samples,
            sha256=h.hexdigest(),
            bytes=(out / "bc-shards.tar.gz").stat().st_size,
        ),
        indent=2,
    )
)
print(json.dumps(dict(episodes=episodes, samples=samples, sha256=h.hexdigest())))
