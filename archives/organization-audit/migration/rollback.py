#!/usr/bin/env python3
"""Reverse the top-level reorganization: remove compat symlinks, move canonical
files/dirs back to their original paths, restore the pre-migration originals of
the management-edited files, and re-point experiments/simulation.

Local filesystem only. No network / Docker / SSH. Verifies each restored
management file against its baseline SHA-256 before overwriting.
"""
import hashlib
import json
import os
import sys

ROOT = "/home/aiden/Desktop/lab/robot/robocasa-docker"
AUDIT = os.path.join(ROOT, "archives", "organization-audit")
MOVEMAP = os.path.join(AUDIT, "migration", "movemap.json")
BASELINE = os.path.join(AUDIT, "inventory-v2.json")
PREMIG = os.path.join(AUDIT, "migration", "pre-migration")

# old-path -> baseline sha256, for management files restored from pre-migration/
RESTORE = ["run.sh", "verify_results.py", "Dockerfile", ".dockerignore", "DIRECTORY_GUIDE.md"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    mm = json.load(open(MOVEMAP, encoding="utf-8"))
    base = json.load(open(BASELINE, encoding="utf-8"))
    report = []

    # 1. remove compat symlinks at old paths
    for cs in mm["compat_symlinks"]:
        link = os.path.join(ROOT, cs["link"])
        if os.path.islink(link):
            os.remove(link)
            report.append(("unlink", cs["link"]))

    # 2. move canonical back to original path
    for mv in mm["moves"]:
        dst = os.path.join(ROOT, mv["to"])
        src = os.path.join(ROOT, mv["from"])
        if os.path.exists(src) and not os.path.islink(src):
            report.append(("skip-exists", mv["from"]))
            continue
        if not os.path.exists(dst):
            report.append(("skip-missing", mv["to"]))
            continue
        os.rename(dst, src)
        report.append(("restore", mv["from"]))

    # 3. restore management-edited originals from pre-migration backups
    for name in RESTORE:
        bak = os.path.join(PREMIG, name)
        tgt = os.path.join(ROOT, name)
        want = base["files"][name]["sha256"]
        if sha256(bak) != want:
            sys.exit(f"rollback abort: backup {name} does not match baseline sha256")
        with open(bak, "rb") as f:
            data = f.read()
        with open(tgt, "wb") as f:
            f.write(data)
        assert sha256(tgt) == want
        report.append(("restore-original", name))

    # 4. re-point experiments/simulation back to ../output
    for rp in mm.get("repoint_symlinks", []):
        link = os.path.join(ROOT, rp["link"])
        if os.path.islink(link):
            os.remove(link)
            os.symlink(rp["old_target"], link)
            report.append(("repoint-back", rp["link"]))

    json.dump({"actions": report}, sys.stdout, indent=1, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
