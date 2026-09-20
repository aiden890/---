from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_source_manifest.py"


def test_manifest_reads_exact_commit_not_dirty_worktree():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
        src = root / "project" / "src" / "x.py"
        src.parent.mkdir(parents=True)
        src.write_text("one\n")
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True)
        src.write_text("two\n")
        out = root / "manifest.json"
        subprocess.run(["python3", str(SCRIPT), "project", str(out), "--commit", "HEAD"],
                       cwd=root, check=True)
        data = json.loads(out.read_text())
        assert data["dirty"] is False
        assert data["files"]["src/x.py"] == hashlib.sha256(b"one\n").hexdigest()

        nested = root / "nested" / "launcher"
        nested.mkdir(parents=True)
        nested_out = root / "nested-manifest.json"
        subprocess.run(["python3", str(SCRIPT), "project", str(nested_out),
                        "--commit", "HEAD"], cwd=nested, check=True)
        nested_data = json.loads(nested_out.read_text())
        assert nested_data["files"] == data["files"]


def test_manifest_excludes_itself_when_output_is_inside_project():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
        project = root / "project"
        project.mkdir()
        (project / "x.py").write_text("one\n")
        (project / "source_manifest.json").write_text("{}\n")
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True)
        subprocess.run(["python3", str(SCRIPT), "project", "project/source_manifest.json",
                        "--commit", "HEAD"], cwd=root, check=True)
        data = json.loads((project / "source_manifest.json").read_text())
        assert "x.py" in data["files"]
        assert "source_manifest.json" not in data["files"]


if __name__ == "__main__":
    test_manifest_reads_exact_commit_not_dirty_worktree()
    test_manifest_excludes_itself_when_output_is_inside_project()
    print("build source manifest tests passed")
