"""Read-only host/process sampling for the training run and its W&B writer."""

import argparse
import json
import subprocess
import time
from pathlib import Path
from online_bc.data.data_control import atomic_json


def remote(host, script):
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, "python3 -"],
        input=script,
        text=True,
        capture_output=True,
        timeout=25,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-500:])
    return json.loads(result.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    c = json.loads(Path(args.config).read_text())
    root = Path(c["root"])
    while True:
        state = dict(at=time.time(), hosts={})
        for node, worker in c["workers"].items():
            config = worker["worker_config"]
            # The Lab copy contains a different home prefix; actual remote
            # config is the --config argument already present in collect_argv.
            command = worker["collect_argv"][-1]
            config = command.split("--config ")[1].split()[0]
            script = f"""
import json, pathlib, urllib.request, subprocess, time
c=json.loads(pathlib.Path({config!r}).read_text())
with urllib.request.urlopen(c["policy_url"],timeout=5) as response: policy=json.load(response)
logs=list(pathlib.Path(c["output_root"]).rglob("worker-*.log"))+list(pathlib.Path(c["output_root"]).parent.glob("evaluation/**/worker-*.log"))
recent=sorted(logs,key=lambda p:p.stat().st_mtime,reverse=True)[:2]
gpu=subprocess.run(["nvidia-smi","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True,capture_output=True)
print(json.dumps(dict(policy=policy,logs=[dict(path=str(p),age_seconds=time.time()-p.stat().st_mtime,tail=p.read_text()[-1000:]) for p in recent],gpu=gpu.stdout.strip())))
"""
            try:
                state["hosts"][node] = dict(ok=True, **remote(worker["host"], script))
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                state["hosts"][node] = dict(ok=False, error=str(exc))
        learner_root = c["learner_root"]
        script = f"""
import json, pathlib, os, subprocess, time
root=pathlib.Path({learner_root!r})
pid=int((root/"learner.pid").read_text()); alive=True
try: os.kill(pid,0)
except ProcessLookupError: alive=False
status=root/"run/service-status.json"
gpu=subprocess.run(["nvidia-smi","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True,capture_output=True)
print(json.dumps(dict(alive=alive,pid=pid,status=json.loads(status.read_text()) if status.exists() else None,gpu=gpu.stdout.strip(),log_tail=(root/"learner.log").read_text()[-1500:])))
"""
        try:
            state["hosts"]["learner"] = dict(ok=True, **remote(c["learner_host"], script))
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            state["hosts"]["learner"] = dict(ok=False, error=str(exc))
        for name in ["coordinator", "telemetry"]:
            try:
                pid = int((root / f"{name}.pid").read_text())
                subprocess.run(["kill", "-0", str(pid)], check=True, capture_output=True)
                state[name] = dict(alive=True, pid=pid)
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                state[name] = dict(alive=False, error=str(exc))
        atomic_json(root / "health.json", state)
        with (root / "health-history.jsonl").open("a") as log:
            log.write(json.dumps(state) + "\n")
        time.sleep(60)


if __name__ == "__main__":
    main()
