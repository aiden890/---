#!/usr/bin/env python3
"""Print the exact source commit from a deployment manifest."""
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
commit = manifest.get("commit")
if not isinstance(commit, str) or len(commit) != 40:
    raise SystemExit("deployment manifest has no exact 40-character commit")
print(commit)
