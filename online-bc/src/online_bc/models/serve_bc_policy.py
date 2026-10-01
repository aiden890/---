"""Loopback-only inference endpoint with explicit checkpoint reloads."""

import argparse
import importlib
import json
import pickle
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["xiaomi", "pi05", "groot"])
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--adapter")
ap.add_argument("--port", type=int, default=18317)
ap.add_argument("--ready-file")
ap.add_argument("--enable-base-prefix", action="store_true")
args = ap.parse_args()
backend = importlib.import_module("online_bc.models." + args.model + "_backend").Backend(
    args.checkpoint
)
version = 0
phase_policy = None
if args.enable_base_prefix:
    if args.model != "pi05":
        raise ValueError("base_prefix currently supports only pi05")
    from openpi.shared import nnx_utils
    from online_bc.models.phase_policy import PhasePolicy

    phase_policy = PhasePolicy(backend, nnx_utils.module_jit(backend.model.sample_actions))
if args.adapter:
    (phase_policy or backend).load(args.adapter)
    version = json.loads((Path(args.adapter) / "metadata.json").read_text())["step"]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        data = json.dumps(
            dict(
                model=args.model,
                version=version,
                ready=True,
                base_prefix_enabled=phase_policy is not None,
            )
        ).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        global version
        try:
            req = pickle.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if req.get("op") == "load":
                path = Path(req["path"])
                metadata = json.loads((path / "metadata.json").read_text())
                assert metadata["model"] == args.model
                (phase_policy or backend).load(path)
                version = metadata["step"]
                result = dict(version=version, loaded=True)
            else:
                variant = req.get("variant", "standard")
                phase = req.get("skill_phase")
                use_base = False
                if phase_policy is not None:
                    prediction, use_base = phase_policy.infer(
                        req["sample"], req.get("seed", 0), variant, phase
                    )
                else:
                    if variant != "standard":
                        raise ValueError("Requested variant is not enabled on this server")
                    prediction = backend.infer(req["sample"], req.get("seed", 0))
                action = np.asarray(prediction, np.float32)
                assert (
                    action.ndim == 2
                    and action.shape[1] == 12
                    and len(action) >= 16
                    and np.isfinite(action).all()
                )
                result = dict(
                    actions=action,
                    version=version,
                    variant=variant,
                    skill_phase=phase,
                    effective_policy_version=0 if use_base else version,
                )
            data = pickle.dumps(result, protocol=4)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(data)
        except Exception as ex:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(ex).encode())
            raise


if args.ready_file:
    Path(args.ready_file).write_text(
        json.dumps(dict(model=args.model, port=args.port, version=version))
    )
print(
    json.dumps(dict(event="READY", model=args.model, port=args.port, version=version)), flush=True
)
HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
