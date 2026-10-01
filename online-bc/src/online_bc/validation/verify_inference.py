"""Verify published learner weights in a fresh rollout process/GPU context."""

import argparse
import importlib
import json
from pathlib import Path
import numpy as np
from online_bc.data.replay import Replay

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--data", nargs="+", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
out = Path(args.out)
report = json.loads((out / "verification.json").read_text())
assert (
    report["finite_loss_and_grad"]
    and report["adapter_changed"]
    and report["checkpoint_reload_equal"]
)
replay = Replay(args.data)
assert replay.ingest() > 0
backend = importlib.import_module("online_bc.models." + args.model + "_backend").Backend(
    args.checkpoint
)
adapter = Path((out / "latest").read_text().strip())
backend.load(adapter)
a = np.asarray(backend.infer(replay.sample("cup_placement"), seed=42))
assert a.ndim == 2 and a.shape[1] == 12 and len(a) >= 16 and np.isfinite(a).all()
report.update(
    passed=True,
    status="passed",
    inference_after_reload=True,
    inference_action_shape=list(a.shape),
    inference_process="fresh rollout process, disaggregated from gradient worker",
)
(out / "verification.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
