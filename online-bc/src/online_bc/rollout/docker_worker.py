"""Use the exact GPU-tested image, mounts, and environment for each worker."""

from online_bc.paths import PROJECT_ROOT

import argparse
import json
import subprocess


def run_args(t):
    argv = ["docker", "run", "--network", "host", "--ipc", "host", "--gpus", "all"]
    for m in t["Mounts"]:
        src = m.get("Name") if m["Type"] == "volume" else m["Source"]
        argv += ["-v", src + ":" + m["Destination"] + ("" if m["RW"] else ":ro")]
    for e in t["Config"]["Env"]:
        argv += ["-e", e]
    argv += [
        "-e",
        "PYTHONPATH=/results/coffee-online-bc/src:"
        + next((e.split("=", 1)[1] for e in t["Config"]["Env"] if e.startswith("PYTHONPATH=")), ""),
    ]
    return argv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("action", choices=["ensure", "collect", "preflight", "dataset"])
    p.add_argument("--seeds")
    p.add_argument("--out")
    p.add_argument("--source")
    a = p.parse_args()
    root = PROJECT_ROOT
    t = json.loads((root / "configs" / f"{a.model}-containers.json").read_text())
    if a.action == "ensure":
        name = f"coffee-online-bc-policy-{a.model}"
        state = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name], capture_output=True, text=True
        )
        if state.returncode == 0 and state.stdout.strip() == "true":
            return
        if state.returncode == 0:
            subprocess.run(["docker", "rm", name], check=True)
        python = t[0]["Config"]["Cmd"][0]
        cmd = t[0]["Config"]["Cmd"]
        checkpoint = cmd[cmd.index("--checkpoint") + 1]
        adapter = []
        current = root / "current-adapter.json"
        if current.exists():
            adapter = ["--adapter", json.loads(current.read_text())["container_path"]]
        subprocess.run(
            run_args(t[0])
            + [
                "-d",
                "--name",
                name,
                "-w",
                "/results/coffee-online-bc/src",
                t[0]["Config"]["Image"],
                python,
                "-m",
                "online_bc.models.serve_bc_policy",
                "--model",
                a.model,
                "--checkpoint",
                checkpoint,
                *adapter,
            ],
            check=True,
        )
    elif a.action == "dataset":
        argv = run_args(t[1]) + [
            "--rm",
            "-w",
            "/results/coffee-online-bc/src",
            t[1]["Config"]["Image"],
        ]
        subprocess.run(
            argv + ["python3", "-m", "online_bc.data.build_cup_dataset", a.source, a.out],
            check=True,
        )
        subprocess.run(
            argv
            + [
                "python3",
                "-m",
                "online_bc.data.validate_dataset",
                a.out,
                "--source-required",
                "--out",
                a.out + "/validation.json",
            ],
            check=True,
        )
    else:
        argv = run_args(t[1]) + ["--rm", "-e", "COFFEE_BC_POLICY_URL=http://127.0.0.1:18317"]
        module = (
            "online_bc.rollout.preflight_scenes"
            if a.action == "preflight"
            else "online_bc.rollout.run_coffee"
        )
        tail = ["--seeds", a.seeds or ",".join(str(993101 + i) for i in range(8))]
        if a.action == "collect":
            tail += (
                ["--model", a.model, "--out", a.out, "--skill", "cup_placement"]
                if a.model == "pi05"
                else ["--model", a.model, "--out", a.out]
            )
        subprocess.run(argv + [t[1]["Config"]["Image"], "python3", "-m", module, *tail], check=True)


if __name__ == "__main__":
    main()
