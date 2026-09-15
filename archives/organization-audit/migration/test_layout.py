#!/usr/bin/env python3
"""Layout acceptance test for the RoboCasa workspace top-level reorganization.

Read-only. No network / Docker / SSH. Exit 0 = all invariants hold (GREEN);
exit 1 = one or more failed (RED). Run before migration (expected RED) and
after (expected GREEN).

Invariants:
  A. Canonical target files/dirs exist at their new locations.
  B. Every old top-level path still resolves (compat symlink -> existing file).
  C. Every symlink under the tree (excl. vendor) resolves to an existing target.
  D. Every Markdown relative link resolves to an existing path.
  E. The set of real .mp4 videos reachable through videos/ equals the baseline set.
  F. `bash -n` passes on every tracked shell script.
  G. Docker build-context COPY sources exist relative to the context root.
  H. Preserved (hashed) files keep their baseline SHA-256 (management-code
     files intentionally changed are listed in EXPECTED_CHANGED and skipped).
"""
import hashlib
import json
import os
import subprocess
import sys

ROOT = "/home/aiden/Desktop/lab/robot/robocasa-docker"
AUDIT = os.path.join(ROOT, "archives", "organization-audit")
BASELINE = os.path.join(AUDIT, "inventory-v2.json")
MOVEMAP = os.path.join(AUDIT, "migration", "movemap.json")

# Files whose content is intentionally edited by the migration (management code).
# Keyed by BASELINE (pre-migration) relpath, because check H iterates baseline
# entries. Their bytes change; excluded from the hash-preservation check H.
# VERIFICATION.md / XIAOMI_STATUS.md move byte-identical (no edits) and are
# therefore NOT listed: H verifies their hash survives the move.
EXPECTED_CHANGED = {
    "run.sh", "verify_results.py",
    "Dockerfile", ".dockerignore", "DIRECTORY_GUIDE.md",
}
SKIP_DIRS = {"__pycache__", ".git", "vendor"}

failures = []
def check(cond, msg):
    if not cond:
        failures.append(msg)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    baseline = load(BASELINE)
    movemap = load(MOVEMAP)

    # A. canonical targets exist
    for mv in movemap["moves"]:
        p = os.path.join(ROOT, mv["to"])
        check(os.path.exists(p) and not os.path.islink(p),
              f"A: canonical target missing or is a symlink: {mv['to']}")

    # B. old paths still resolve (via compat symlink)
    for cs in movemap["compat_symlinks"]:
        lp = os.path.join(ROOT, cs["link"])
        check(os.path.islink(lp), f"B: expected compat symlink at old path: {cs['link']}")
        check(os.path.exists(lp), f"B: compat symlink does not resolve: {cs['link']}")

    # C. every symlink resolves
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in list(dirnames) + filenames:
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                check(os.path.exists(full),
                      f"C: dangling symlink: {os.path.relpath(full, ROOT)} -> {os.readlink(full)}")

    # D. markdown relative links resolve
    import re
    link_re = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if not name.endswith(".md"):
                continue
            full = os.path.join(dirpath, name)
            # Skip compat symlinks (the real file is checked at its canonical
            # location) and archived pre-migration doc snapshots (historical
            # copies whose links are relative to the old layout on purpose).
            if os.path.islink(full):
                continue
            if "organization-audit/migration/pre-migration" in full.replace(os.sep, "/"):
                continue
            try:
                text = open(full, encoding="utf-8").read()
            except Exception:
                continue
            for m in link_re.findall(text):
                url = m.split()[0].strip()
                if "://" in url or url.startswith("#") or url.startswith("mailto:"):
                    continue
                target = url.split("#")[0]
                if not target:
                    continue
                resolved = os.path.normpath(os.path.join(dirpath, target))
                check(os.path.exists(resolved),
                      f"D: broken md link in {os.path.relpath(full, ROOT)}: {url}")

    # E. video set equality vs baseline (real mp4 files reachable through videos/)
    def video_realpaths(base):
        vids = set()
        vroot = os.path.join(base, "videos")
        for dp, dn, fn in os.walk(vroot):
            for n in fn:
                if n.endswith(".mp4"):
                    vids.add(os.path.realpath(os.path.join(dp, n)))
        return vids
    # Baseline video set = resolved symlink targets under videos/ UNION real
    # .mp4 files recorded directly under videos/ (worker-added, not symlinked).
    base_videos = {os.path.realpath(os.path.join(ROOT, rp))
                   for rp in baseline["symlinks"]
                   if rp.startswith("videos/") and rp.endswith(".mp4")}
    base_videos |= {os.path.realpath(os.path.join(ROOT, rp))
                    for rp in baseline["files"]
                    if rp.startswith("videos/") and rp.endswith(".mp4")}
    now_videos = video_realpaths(ROOT)
    check(base_videos == now_videos,
          f"E: video set changed. missing={sorted(base_videos-now_videos)} extra={sorted(now_videos-base_videos)}")

    # F. bash -n on all shell scripts
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.endswith(".sh"):
                full = os.path.join(dirpath, name)
                if os.path.islink(full):
                    continue
                r = subprocess.run(["bash", "-n", full], capture_output=True, text=True)
                check(r.returncode == 0, f"F: bash -n failed: {os.path.relpath(full, ROOT)}: {r.stderr.strip()}")

    # G. docker COPY sources exist relative to context root (ROOT)
    for src in movemap["docker_copy_sources_required"]:
        check(os.path.exists(os.path.join(ROOT, src)),
              f"G: Docker COPY source missing: {src}")

    # H. preserved files keep baseline sha256
    for rp, meta in baseline["files"].items():
        if "sha256" not in meta:
            continue
        if rp in EXPECTED_CHANGED:
            continue
        # Skip the audit workspace's own generated artifacts (inventory, movemap,
        # test outputs): these are management files created/updated by this task,
        # not preserved originals.
        if rp.startswith("archives/organization-audit/"):
            continue
        # locate current path: either same rp, or moved to new location
        moved_to = None
        for mv in movemap["moves"]:
            if rp == mv["from"] or rp.startswith(mv["from"] + "/"):
                moved_to = mv["to"] + rp[len(mv["from"]):]
                break
        cur = os.path.join(ROOT, moved_to if moved_to else rp)
        if not os.path.exists(cur):
            failures.append(f"H: preserved file vanished: {rp}")
            continue
        if os.path.realpath(cur) != cur and os.path.islink(cur):
            cur = os.path.realpath(cur)
        got = sha256(cur)
        check(got == meta["sha256"],
              f"H: sha256 changed for preserved file {rp} (now via {moved_to or rp})")

    if failures:
        print(f"RED: {len(failures)} invariant failure(s):")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("GREEN: all layout invariants hold.")
    sys.exit(0)


if __name__ == "__main__":
    main()
