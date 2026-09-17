#!/usr/bin/env python3
"""Build a source manifest from an exact Git commit, never the dirty worktree."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def run(*args: str, text: bool = True):
    return subprocess.check_output(args, text=text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("project", help="Repo-relative deployed directory")
    ap.add_argument("output")
    ap.add_argument("--commit", default="HEAD")
    args = ap.parse_args()

    repo = Path(run("git", "rev-parse", "--show-toplevel").strip())
    project = args.project.rstrip("/")
    commit = run("git", "rev-parse", args.commit).strip()
    names = run("git", "ls-tree", "-r", "--name-only", commit, "--", project).splitlines()
    selected: list[str] = []
    for name in names:
        rel = name[len(project) + 1:]
        if not rel or rel.startswith(("results/", "REPORT/", "vendor/", "data/", "pylibs/")):
            continue
        if rel.endswith((".pt", ".mp4", ".tar.gz")) or "/__pycache__/" in f"/{rel}/":
            continue
        selected.append(rel)
    if not selected:
        raise SystemExit(f"no tracked deployable files under {project}")

    file_hashes: dict[str, str] = {}
    manifest: dict[str, object] = {
        "commit": commit,
        "dirty": False,
        "build_worktree_dirty": bool(run("git", "status", "--porcelain").strip()),
        "project": project,
        "files": file_hashes,
    }
    for rel in sorted(selected):
        blob = run("git", "show", f"{commit}:{project}/{rel}", text=False)
        file_hashes[rel] = hashlib.sha256(blob).hexdigest()
    Path(args.output).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
