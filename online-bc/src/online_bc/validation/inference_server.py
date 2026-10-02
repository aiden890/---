"""Isolated real-GPU HTTP validation; never changes a production policy server."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
import threading
import time
import urllib.request
from http.server import HTTPServer

import numpy as np

from online_bc.data.data_control import atomic_json
from online_bc.data.replay import Replay
from online_bc.models.pi05_backend import Backend
from online_bc.models.serve_bc_policy import PolicyService, make_handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapters", nargs=2, required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    singleton = (output / "validation.lock").open("a")
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report_path = output / "report.json"
    if report_path.exists() and json.loads(report_path.read_text()).get("status") == "passed":
        raise SystemExit("Completed validation already exists; refusing duplicate")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    locks = Path.home() / ".cache/coffee-online-bc-gpu-locks"
    locks.mkdir(parents=True, exist_ok=True)
    lock = (locks / f"gpu-{hashlib.sha256(gpu.encode()).hexdigest()[:16]}.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        atomic_json(report_path, dict(status="deferred_gpu_busy", deployed=False))
        return
    atomic_json(output / "started.json", dict(pid=os.getpid(), time=time.time()))
    started = time.monotonic()
    replay = Replay(args.data, seed=5335)
    assert replay.ingest() > 0
    samples = [replay.sample("cup_placement") for _ in range(3)]
    seeds = [533501, 533502, 533503]
    backend = Backend(args.checkpoint)
    base_actions = [backend.infer(s, seed).astype(np.float32) for s, seed in zip(samples, seeds)]
    service = PolicyService("pi05", backend, inference_only=True, enable_base_prefix=True)
    server = HTTPServer(("127.0.0.1", 0), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"

    def request(payload):
        req = urllib.request.Request(url, data=pickle.dumps(payload, protocol=4))
        with urllib.request.urlopen(req, timeout=180) as response:
            assert response.status == 200
            return pickle.loads(response.read())

    rows = []
    try:
        for adapter in args.adapters:
            # Full training restore supplies an independent action reference; no updates.
            backend.load(adapter)
            reference = [backend.infer(s, seed).astype(np.float32)
                         for s, seed in zip(samples, seeds)]
            optimizer = backend.opt_state
            version = json.loads((Path(adapter) / "metadata.json").read_text())["step"]
            with tempfile.TemporaryDirectory(prefix="http-adapter-only-", dir=output) as tmp:
                path = Path(tmp)
                for name in ["metadata.json", "pi05-lora.npz", "updates.json"]:
                    (path / name).symlink_to((Path(adapter) / name).resolve())
                assert not (path / "pi05-optimizer.pkl").exists()
                loaded = request(dict(op="load", path=str(path)))
                assert loaded["version"] == version and loaded["inference_only"] is True
                assert service.phase_policy.adapter_snapshot is None
                assert backend.opt_state is optimizer and backend.inference_only
                health = json.load(urllib.request.urlopen(url, timeout=10))
                assert health["version"] == version and health["inference_only"]
                for i, (sample, seed) in enumerate(zip(samples, seeds)):
                    for variant, phase in [("standard", None), ("base_prefix", "prefix"),
                                           ("base_prefix", "cup_placement"), ("standard", None)]:
                        payload = dict(sample=sample, seed=seed)
                        if variant != "standard":
                            payload.update(variant=variant, skill_phase=phase)
                        result = request(payload)
                        is_base = phase == "prefix"
                        expected = base_actions[i] if is_base else reference[i]
                        assert np.isfinite(result["actions"]).all()
                        assert np.array_equal(result["actions"], expected)
                        assert result["version"] == version and result["variant"] == variant
                        assert result["skill_phase"] == phase
                        assert result["effective_policy_version"] == (0 if is_base else version)
                    rows.append(dict(version=version, episode_id=sample["episode_id"],
                                     step=sample["step"], rng_seed=seed, max_abs_error=0.0))
                try:
                    backend.update_batch([])
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("Inference-only training guard failed")
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()
    report = dict(status="passed", deployed=False, samples=rows,
                  inference_requests=24, load_requests=2,
                  optimizer_missing=True, initial_and_reload_verified=True,
                  base_prefix_cache_and_interleaving_verified=True,
                  production_updates=0, live_policy_changes=False,
                  wall_seconds=time.monotonic() - started,
                  scope="Same-process HTTP action/receipt equivalence; no rollout or network speed claim")
    atomic_json(report_path, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
