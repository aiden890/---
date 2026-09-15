#!/usr/bin/env python3
"""Post-migration integrity diff: compare current bytes to the pre-migration
baseline (inventory-v2.json) for every originally-hashed file, following the
movemap so relocated files are hashed at their new canonical path. Emits the
exact set of changed / vanished files and asserts it is a subset of the declared
management-edit set. Read-only; stdlib only."""
import hashlib
import json
import os
import sys

ROOT = "/home/aiden/Desktop/lab/robot/robocasa-docker"
AUDIT = os.path.join(ROOT, "archives", "organization-audit")
BASE = json.load(open(os.path.join(AUDIT, "inventory-v2.json"), encoding="utf-8"))
MM = json.load(open(os.path.join(AUDIT, "migration", "movemap.json"), encoding="utf-8"))

# Declared, intentional management-code edits (baseline relpaths).
DECLARED_CHANGED = {"run.sh", "verify_results.py", "Dockerfile", ".dockerignore", "DIRECTORY_GUIDE.md"}
# The audit workspace's own generated/edited files are not "preserved originals".
IGNORE_PREFIX = "archives/organization-audit/"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current_path(rp):
    for mv in MM["moves"]:
        if rp == mv["from"] or rp.startswith(mv["from"] + "/"):
            return os.path.join(ROOT, mv["to"] + rp[len(mv["from"]):])
    return os.path.join(ROOT, rp)


changed, vanished, preserved = [], [], 0
for rp, meta in BASE["files"].items():
    if "sha256" not in meta or rp.startswith(IGNORE_PREFIX):
        continue
    cur = current_path(rp)
    if not os.path.exists(cur):
        vanished.append(rp)
        continue
    if sha256(os.path.realpath(cur)) == meta["sha256"]:
        preserved += 1
    else:
        changed.append(rp)

report = {
    "preserved_byte_identical": preserved,
    "changed": sorted(changed),
    "vanished": sorted(vanished),
    "declared_changed": sorted(DECLARED_CHANGED),
    "undeclared_changes": sorted(set(changed) - DECLARED_CHANGED),
    "vanished_count": len(vanished),
}
print(json.dumps(report, indent=2, ensure_ascii=False))
if report["undeclared_changes"] or report["vanished"]:
    sys.exit(1)
