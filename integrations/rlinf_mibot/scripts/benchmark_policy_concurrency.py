#!/usr/bin/env python3
"""Short policy-server concurrency benchmark using a saved CloseBlenderLid observation."""
from __future__ import annotations

import argparse
import json
import pickle
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

for _path in ("/train/src", "/work", "/rl_env/src"):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import rollout
from grpo_train_loop import TrainerClient


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            raise EOFError("policy server closed the connection")
        data.extend(packet)
    return bytes(data)


class RawClient:
    def __init__(self, host: str, port: int):
        self.sock = socket.create_connection((host, port), timeout=120)

    def call(self, request: dict) -> dict:
        blob = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        self.sock.sendall(struct.pack(">I", len(blob)) + blob)
        size = struct.unpack(">I", recv_exact(self.sock, 4))[0]
        response = pickle.loads(recv_exact(self.sock, size))
        if response.get("error"):
            raise RuntimeError(response["error"])
        return response

    def close(self):
        self.sock.close()


def gpu_sampler(stop: threading.Event, samples: list[int]):
    while not stop.wait(0.2):
        try:
            value = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                text=True, timeout=2).splitlines()[0]
            samples.append(int(value.strip()))
        except Exception:
            pass


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def run_level(host: str, port: int, inputs: dict, version: int,
              concurrency: int, repeats: int, target_rps_per_client: float | None) -> dict:
    barrier = threading.Barrier(concurrency)
    latencies: list[float] = []
    lock = threading.Lock()
    gpu_samples: list[int] = []
    stop = threading.Event()
    sampler = threading.Thread(target=gpu_sampler, args=(stop, gpu_samples), daemon=True)

    def worker(worker_id: int):
        client = RawClient(host, port)
        local = []
        try:
            barrier.wait()
            next_start = time.perf_counter()
            for repeat in range(repeats):
                if target_rps_per_client:
                    delay = next_start - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                started = time.perf_counter()
                response = client.call({
                    "op": "sample", "inputs": inputs, "eta": 0.0,
                    "traj_id": f"bench-{concurrency}-{worker_id}-{repeat}",
                    "seed": 7000 + worker_id, "skill": "grasp", "chunk_index": repeat,
                    "expected_policy_version": version,
                })
                elapsed = time.perf_counter() - started
                if not bool(torch.isfinite(response["actions"]).all()):
                    raise RuntimeError("non-finite action returned")
                local.append(elapsed)
                if target_rps_per_client:
                    next_start += 1.0 / target_rps_per_client
        finally:
            client.close()
        with lock:
            latencies.extend(local)

    sampler.start()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(worker, range(concurrency)))
    wall = time.perf_counter() - started
    stop.set()
    sampler.join(timeout=2)
    return {
        "concurrency": concurrency,
        "requests": len(latencies),
        "wall_seconds": wall,
        "requests_per_second": len(latencies) / wall,
        "target_rps_per_client": target_rps_per_client,
        "latency_mean_seconds": statistics.mean(latencies),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "gpu_util_mean_percent": (statistics.mean(gpu_samples) if gpu_samples else None),
        "gpu_util_max_percent": (max(gpu_samples) if gpu_samples else None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", required=True)
    parser.add_argument("--model-path", default="/checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10088)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--target-rps-per-client", type=float)
    parser.add_argument("--output")
    args = parser.parse_args()

    obs = dict(np.load(args.observation))
    state = rollout.observation_to_state(obs)
    states = [state] * 4
    images = {key: [obs[key]] * 4 for key in rollout.CAMERA_KEYS}
    builder = TrainerClient(args.model_path, args.host, args.port, "robocasa365", 0.95)
    try:
        metrics = builder.metrics()
        inputs = builder._build_inputs(
            states, images, "Close the lid blender by securely placing the lid on top.")
    finally:
        builder.close()

    # One unmeasured request initializes kernels and allocator caches.
    warm = RawClient(args.host, args.port)
    try:
        warm.call({"op": "sample", "inputs": inputs, "eta": 0.0, "seed": 7,
                   "skill": "grasp", "chunk_index": 0,
                   "expected_policy_version": int(metrics["policy_version"])})
    finally:
        warm.close()

    report = {
        "policy_version": int(metrics["policy_version"]),
        "policy_hash": metrics["policy_hash"],
        "repeats_per_client": args.repeats,
        "target_rps_per_client": args.target_rps_per_client,
        "levels": [run_level(args.host, args.port, inputs, int(metrics["policy_version"]),
                             level, args.repeats, args.target_rps_per_client)
                   for level in args.concurrency],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
