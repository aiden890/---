"""Loopback-only inference endpoint with explicit checkpoint reloads."""

import argparse
import importlib
import json
import pickle
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np


class PolicyService:
    def __init__(self, model, backend, *, inference_only=False, enable_base_prefix=False):
        if model != "pi05" and (inference_only or enable_base_prefix):
            raise ValueError("inference_only and base_prefix currently support only pi05")
        self.model = model
        self.backend = backend
        self.inference_only = inference_only
        self.version = 0
        self.phase_policy = None
        if enable_base_prefix:
            from openpi.shared import nnx_utils
            from online_bc.models.phase_policy import PhasePolicy

            # Capture zero-B inference before loading any trained adapter.
            self.phase_policy = PhasePolicy(
                backend, nnx_utils.module_jit(backend.model.sample_actions)
            )

    def health(self):
        return dict(model=self.model, version=self.version, ready=True,
                    base_prefix_enabled=self.phase_policy is not None,
                    inference_only=self.inference_only)

    def load(self, path):
        metadata = json.loads((Path(path) / "metadata.json").read_text())
        assert metadata["model"] == self.model
        policy = self.phase_policy or self.backend
        if self.inference_only:
            policy.load(path, load_optimizer=False)
        else:
            policy.load(path)
        self.version = metadata["step"]
        return dict(version=self.version, loaded=True, inference_only=self.inference_only)

    def request(self, req):
        if req.get("op") == "load":
            return self.load(req["path"])
        variant = req.get("variant", "standard")
        phase = req.get("skill_phase")
        use_base = False
        if self.phase_policy is not None:
            prediction, use_base = self.phase_policy.infer(
                req["sample"], req.get("seed", 0), variant, phase
            )
        else:
            if variant != "standard":
                raise ValueError("Requested variant is not enabled on this server")
            prediction = self.backend.infer(req["sample"], req.get("seed", 0))
        action = np.asarray(prediction, np.float32)
        assert (action.ndim == 2 and action.shape[1] == 12
                and len(action) >= 16 and np.isfinite(action).all())
        return dict(actions=action, version=self.version, variant=variant, skill_phase=phase,
                    effective_policy_version=0 if use_base else self.version)


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(service.health()).encode())

        def do_POST(self):
            try:
                req = pickle.loads(self.rfile.read(int(self.headers["Content-Length"])))
                data = pickle.dumps(service.request(req), protocol=4)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(data)
            except Exception as ex:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(ex).encode())
                raise

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["xiaomi", "pi05", "groot"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--port", type=int, default=18317)
    ap.add_argument("--ready-file")
    ap.add_argument("--enable-base-prefix", action="store_true")
    ap.add_argument("--inference-only", action="store_true")
    args = ap.parse_args()
    if args.model != "pi05" and (args.inference_only or args.enable_base_prefix):
        ap.error("inference_only and base_prefix currently support only pi05")
    backend = importlib.import_module("online_bc.models." + args.model + "_backend").Backend(
        args.checkpoint
    )
    service = PolicyService(args.model, backend, inference_only=args.inference_only,
                            enable_base_prefix=args.enable_base_prefix)
    if args.adapter:
        service.load(args.adapter)
    if args.ready_file:
        Path(args.ready_file).write_text(json.dumps(dict(port=args.port, **service.health())))
    print(json.dumps(dict(event="READY", port=args.port, **service.health())), flush=True)
    HTTPServer(("127.0.0.1", args.port), make_handler(service)).serve_forever()


if __name__ == "__main__":
    main()
