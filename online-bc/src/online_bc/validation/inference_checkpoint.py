"""Compare full and adapter-only loads on real observations, under the learner GPU lock."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from online_bc.data.data_control import atomic_json
from online_bc.data.replay import Replay
from online_bc.models.pi05_backend import Backend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter", required=True)
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
    backend = Backend(args.checkpoint)
    before = time.monotonic()
    backend.load(args.adapter)
    backend.jax.block_until_ready(backend.opt_state)
    backend.jax.block_until_ready(backend.nnx.state(backend.model, backend.filter))
    full_load_seconds = time.monotonic() - before
    seeds = [533501, 533502, 533503]
    reference = [backend.infer(sample, seed) for sample, seed in zip(samples, seeds)]
    original_optimizer = backend.opt_state
    rows = []
    with tempfile.TemporaryDirectory(prefix="adapter-only-", dir=output) as tmp:
        adapter_only = Path(tmp)
        (adapter_only / "pi05-lora.npz").symlink_to(
            (Path(args.adapter) / "pi05-lora.npz").resolve()
        )
        assert not (adapter_only / "pi05-optimizer.pkl").exists()
        before = time.monotonic()
        backend.load(adapter_only, load_optimizer=False)
        backend.jax.block_until_ready(backend.nnx.state(backend.model, backend.filter))
        adapter_load_seconds = time.monotonic() - before
        assert backend.opt_state is original_optimizer and backend.infer_fn is None
        for sample, seed, actions in zip(samples, seeds, reference):
            before = time.monotonic()
            actual = backend.infer(sample, seed)
            elapsed = time.monotonic() - before
            assert np.isfinite(actual).all() and np.array_equal(actions, actual)
            rows.append(dict(episode_id=sample["episode_id"], step=sample["step"],
                             rng_seed=seed, max_abs_error=float(np.max(np.abs(actions - actual))),
                             inference_seconds=elapsed))
        for operation in [lambda: backend.update_batch([]), lambda: backend.save(adapter_only, 0)]:
            try:
                operation()
            except RuntimeError:
                pass
            else:
                raise AssertionError("Inference-only training guard failed")
        try:
            backend.load(adapter_only)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Default training restore accepted missing optimizer")
    report = dict(status="passed", deployed=False, production_updates=0,
                  checkpoint=args.checkpoint, adapter=args.adapter, samples=rows,
                  full_load_seconds=full_load_seconds, adapter_only_load_seconds=adapter_load_seconds,
                  optimizer_object_preserved=True, training_and_save_blocked=True,
                  default_restore_requires_optimizer=True,
                  optimizer_bytes=(Path(args.adapter) / "pi05-optimizer.pkl").stat().st_size,
                  wall_seconds=time.monotonic() - started,
                  scope="Same-process fixed-RNG equivalence; no network transfer or rollout speed measurement",
                  filtered_downloads_deployed=False, live_policy_changes=False)
    atomic_json(report_path, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
