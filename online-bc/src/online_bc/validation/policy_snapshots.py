"""Validate frozen base/adapter inference snapshots without changing live policies."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from online_bc.data.data_control import atomic_json
from online_bc.data.replay import Replay
from online_bc.models.pi05_backend import Backend
from online_bc.models.phase_policy import PhasePolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--wait-for-gpu", action="store_true")
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    locks = Path.home() / ".cache/coffee-online-bc-gpu-locks"
    locks.mkdir(parents=True, exist_ok=True)
    lock = (locks / f"gpu-{hashlib.sha256(gpu.encode()).hexdigest()[:16]}.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        if not args.wait_for_gpu:
            atomic_json(output / "report.json", dict(status="deferred_gpu_busy", deployed=False))
            return
        atomic_json(output / "report.json", dict(status="waiting_for_gpu", deployed=False))
        fcntl.flock(lock, fcntl.LOCK_EX)
    started = time.monotonic()
    replay = Replay(args.data, seed=5335)
    assert replay.ingest() > 0
    samples = [replay.sample("cup_placement") for _ in range(3)]
    backend = Backend(args.checkpoint)
    seeds = [533501, 533502, 533503]
    reference = [backend.infer(s, seed) for s, seed in zip(samples, seeds)]
    base_snapshot = backend.infer_fn
    router = PhasePolicy(backend, base_snapshot)
    router.load(args.adapter)
    adapted = [router.infer(s, seed)[0] for s, seed in zip(samples, seeds)]
    rows = []
    for index, (sample, seed) in enumerate(zip(samples, seeds)):
        # Alternate in the same process; module_jit holds each frozen weight state.
        before = time.monotonic()
        base_again, base_selected = router.infer(sample, seed, "base_prefix", "prefix")
        base_seconds = time.monotonic() - before
        before = time.monotonic()
        adapter_again, adapter_base_selected = router.infer(
            sample, seed, "base_prefix", "cup_placement"
        )
        adapter_seconds = time.monotonic() - before
        assert base_selected and not adapter_base_selected
        assert np.isfinite(base_again).all() and np.isfinite(adapter_again).all()
        assert np.array_equal(reference[index], base_again)
        assert np.array_equal(adapted[index], adapter_again)
        rows.append(
            dict(
                episode_id=sample["episode_id"],
                step=sample["step"],
                rng_seed=seed,
                base_max_abs_error=float(np.max(np.abs(reference[index] - base_again))),
                adapter_max_abs_error=float(np.max(np.abs(adapted[index] - adapter_again))),
                adapter_base_max_abs_difference=float(
                    np.max(np.abs(adapted[index] - reference[index]))
                ),
                base_warm_seconds=base_seconds,
                adapter_warm_seconds=adapter_seconds,
            )
        )
    assert any(row["adapter_base_max_abs_difference"] > 0 for row in rows)
    report = dict(
        status="passed",
        deployed=False,
        checkpoint=args.checkpoint,
        adapter=args.adapter,
        samples=rows,
        wall_seconds=time.monotonic() - started,
        production_updates=0,
        simulator_changes=False,
        scope="Frozen inference snapshot equivalence only; skill routing and rollout performance remain untested.",
        source="https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/shared/nnx_utils.py",
    )
    atomic_json(output / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
