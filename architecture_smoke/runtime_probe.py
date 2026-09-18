#!/usr/bin/env python3
"""Read and strictly validate the combined inference server health endpoint."""
from __future__ import annotations

import argparse
import json
import pickle
import socket
import struct
from pathlib import Path


def recv_all(sock: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            raise ConnectionError("server closed the health connection")
        data += packet
    return data


def health(host: str, port: int, timeout: float) -> dict:
    request = pickle.dumps({"op": "health"}, protocol=pickle.HIGHEST_PROTOCOL)
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(struct.pack(">I", len(request)) + request)
        length = struct.unpack(">I", recv_all(sock, 4))[0]
        response = pickle.loads(recv_all(sock, length))
    if not isinstance(response, dict):
        raise TypeError("health response is not a mapping")
    return response


def validate(response: dict) -> None:
    scheduler = response.get("scheduler")
    if response.get("healthy") is not True or response.get("ready") is not True:
        raise RuntimeError(f"server is not healthy and ready: {response!r}")
    if response.get("model_load_count") != 1:
        raise RuntimeError(f"expected exactly one model load: {response!r}")
    if not isinstance(scheduler, dict) or scheduler.get("ready") is not True:
        raise RuntimeError(f"dispatcher is not ready: {response!r}")
    if scheduler.get("max_active_forwards", 0) > 1:
        raise RuntimeError(f"concurrent CUDA forwards detected: {response!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    response = health(args.host, args.port, args.timeout)
    if args.strict:
        validate(response)
    payload = json.dumps(response, indent=2, sort_keys=True)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(payload + "\n")
        temporary.replace(target)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
