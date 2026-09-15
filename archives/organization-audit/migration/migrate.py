#!/usr/bin/env python3
"""Execute the RoboCasa top-level reorganization from movemap.json.

Local filesystem only. No network / Docker / SSH. For every move it re-hashes
the source, performs an in-filesystem rename, verifies the destination hash is
identical (refuses to overwrite), and leaves a relative compat symlink at the
old path so every prior reference still resolves. Idempotent: already-migrated
entries (old path is a symlink to the new path) are skipped.
"""
import hashlib
import json
import os
import sys

ROOT = "/home/aiden/Desktop/lab/robot/robocasa-docker"
MOVEMAP = os.path.join(ROOT, "archives", "organization-audit", "migration", "movemap.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hashes(path):
    """Map of relpath->sha256 for a file or every regular file under a dir."""
    out = {}
    if os.path.isfile(path) and not os.path.islink(path):
        out["."] = sha256(path)
        return out
    for dp, dn, fn in os.walk(path):
        for n in fn:
            full = os.path.join(dp, n)
            if os.path.islink(full):
                continue
            out[os.path.relpath(full, path)] = sha256(full)
    return out


def relsymlink(link_path, target_path):
    """Create link_path as a symlink to target_path, expressed relatively."""
    rel = os.path.relpath(target_path, os.path.dirname(link_path))
    if os.path.islink(link_path):
        if os.readlink(link_path) == rel:
            return
        raise SystemExit(f"refuse: symlink already exists differently: {link_path}")
    if os.path.exists(link_path):
        raise SystemExit(f"refuse: old path still a real file after move: {link_path}")
    os.symlink(rel, link_path)


def main():
    mm = json.load(open(MOVEMAP, encoding="utf-8"))
    report = {"moves": [], "compat_symlinks": [], "repointed": []}

    for mv in mm["moves"]:
        src = os.path.join(ROOT, mv["from"])
        dst = os.path.join(ROOT, mv["to"])
        # idempotent skip: old path already a symlink and new path exists
        if os.path.islink(src) and os.path.exists(dst):
            report["moves"].append({**mv, "status": "already-migrated"})
            continue
        if not os.path.exists(src):
            raise SystemExit(f"missing source: {mv['from']}")
        if os.path.islink(src):
            raise SystemExit(f"source is already a symlink (unexpected): {mv['from']}")
        if os.path.exists(dst):
            raise SystemExit(f"destination already exists: {mv['to']}")
        before = tree_hashes(src)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.rename(src, dst)
        after = tree_hashes(dst)
        if before != after:
            raise SystemExit(f"hash mismatch after move for {mv['from']}")
        report["moves"].append({**mv, "status": "moved", "files": len(before)})

    # compat symlinks at old paths
    for cs in mm["compat_symlinks"]:
        link = os.path.join(ROOT, cs["link"])
        target = os.path.join(ROOT, cs["target"])
        relsymlink(link, target)
        report["compat_symlinks"].append(cs)

    # repoint internal symlinks
    for rp in mm.get("repoint_symlinks", []):
        link = os.path.join(ROOT, rp["link"])
        cur = os.readlink(link) if os.path.islink(link) else None
        if cur == rp["new_target"]:
            report["repointed"].append({**rp, "status": "already"})
            continue
        if cur != rp["old_target"]:
            raise SystemExit(f"repoint: {rp['link']} points to {cur}, expected {rp['old_target']}")
        os.remove(link)
        os.symlink(rp["new_target"], link)
        report["repointed"].append({**rp, "status": "repointed"})

    json.dump(report, sys.stdout, indent=1, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
