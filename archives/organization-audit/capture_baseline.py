#!/usr/bin/env python3
"""Capture a pre-migration inventory: path/size/mode/sha256 for every regular
file, plus symlink targets, under the project root. Large immutable trees
(vendor, model checkpoint, research model snapshot) are recorded by listing but
NOT hashed. Read-only; no network/Docker/SSH."""
import hashlib
import json
import os
import sys

ROOT = "/home/aiden/Desktop/lab/robot/robocasa-docker"
# Directories whose *contents* we list but do not hash (large / immutable / vendored).
NOHASH_PREFIXES = (
    "vendor",
    "xiaomi-cu121/checkpoint",
    "research/t_1d1b4ded/sources/robocasa/models",
)
SKIP_DIRS = {"__pycache__", ".git"}


def rel(p):
    return os.path.relpath(p, ROOT)


def nohash(relpath):
    return any(relpath == p or relpath.startswith(p + "/") for p in NOHASH_PREFIXES)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    files = {}
    symlinks = {}
    dirs = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        # record symlinked dirs (os.walk does not descend them by default)
        for d in list(dirnames):
            full = os.path.join(dirpath, d)
            if os.path.islink(full):
                symlinks[rel(full)] = os.readlink(full)
                dirnames.remove(d)
        r = rel(dirpath)
        if r != ".":
            dirs.append(r)
        for name in filenames:
            full = os.path.join(dirpath, name)
            rp = rel(full)
            if os.path.islink(full):
                symlinks[rp] = os.readlink(full)
                continue
            try:
                st = os.stat(full)
            except OSError as e:
                files[rp] = {"error": str(e)}
                continue
            entry = {"size": st.st_size, "mode": oct(st.st_mode & 0o777)}
            if not nohash(rp):
                entry["sha256"] = sha256(full)
            files[rp] = entry
    out = {
        "root": ROOT,
        "nohash_prefixes": list(NOHASH_PREFIXES),
        "counts": {
            "files": len(files),
            "hashed": sum(1 for v in files.values() if "sha256" in v),
            "symlinks": len(symlinks),
            "dirs": len(dirs),
        },
        "files": dict(sorted(files.items())),
        "symlinks": dict(sorted(symlinks.items())),
        "dirs": sorted(dirs),
    }
    json.dump(out, sys.stdout, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
