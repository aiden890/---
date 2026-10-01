"""Verify native restored weights through the actual rollout HTTP protocol."""

import json
import pickle
import time
import urllib.request
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from online_bc.data.replay import Replay

p = argparse.ArgumentParser()
p.add_argument("--data", nargs="+", required=True)
p.add_argument("--out", required=True)
p.add_argument("--version", type=int, default=2)
a = p.parse_args()
url = "http://127.0.0.1:18317"
for _ in range(120):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            health = json.load(response)
        break
    except OSError:
        time.sleep(2)
else:
    raise RuntimeError("HTTP policy did not become ready")
assert health["model"] == "pi05" and health["version"] == a.version
r = Replay(a.data)
assert r.ingest() == 235
samples = [r.sample("cup_placement") for _ in range(3)]


def check(pair):
    index, sample = pair
    t = time.monotonic()
    request = urllib.request.Request(
        url, data=pickle.dumps(dict(sample=sample, seed=42 + index), protocol=4)
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        reply = pickle.loads(response.read())
    actions = np.asarray(reply["actions"])
    assert actions.shape == (50, 12) and np.isfinite(actions).all()
    assert reply["version"] == a.version
    return dict(
        index=index,
        shape=list(actions.shape),
        seconds=time.monotonic() - t,
        source_model=sample["source_model"],
    )


with ThreadPoolExecutor(max_workers=3) as pool:
    checks = list(pool.map(check, enumerate(samples)))
out = Path(a.out)
report = json.loads((out / "verification.json").read_text())
assert all(
    report[k]
    for k in [
        "finite_loss_and_grad",
        "adapter_changed",
        "checkpoint_reload_equal",
        "frozen_backbone_unchanged",
        "optimizer_resume_equal",
    ]
)
report.update(
    passed=True,
    status="passed",
    inference_after_reload=True,
    inference_action_shape=[50, 12],
    http_requests=checks,
    parallel_rollout_requests=3,
)
(out / "verification.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
