"""Launch the verified native learner environment on an existing worker host."""

from online_bc.paths import PROJECT_ROOT

import argparse
import json
import subprocess
from online_bc.rollout.docker_worker import run_args

p = argparse.ArgumentParser()
p.add_argument("--model", default="pi05")
p.add_argument("--name", required=True)
p.add_argument("command", choices=["train", "infer"])
p.add_argument("argv", nargs=argparse.REMAINDER)
a = p.parse_args()
t = json.loads((PROJECT_ROOT / "configs" / f"{a.model}-containers.json").read_text())[0]
cmd = t["Config"]["Cmd"]
checkpoint = cmd[cmd.index("--checkpoint") + 1]
module = {
    "train": "online_bc.learning.train_online_bc",
    "infer": "online_bc.validation.verify_inference",
}[a.command]
subprocess.run(
    run_args(t)
    + [
        "-d",
        "--name",
        a.name,
        "-w",
        "/results/coffee-online-bc/src",
        t["Config"]["Image"],
        cmd[0],
        "-m",
        module,
        "--model",
        a.model,
        "--checkpoint",
        checkpoint,
        *a.argv,
    ],
    check=True,
)
