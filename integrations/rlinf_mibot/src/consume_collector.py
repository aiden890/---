#!/usr/bin/env python3
"""Consume collector payloads through the existing GRPO trainer update/save/load path."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import socket
import struct
import time
from pathlib import Path


def load_collector_payloads(run_dir: Path) -> list[dict]:
    run_dir = Path(run_dir).resolve()
    rows = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text().splitlines()
            if line.strip()]
    payloads = []
    for row in rows:
        payload = dict(row["trainer_payload"])
        if payload.get("optimizer_update_requested") is not False:
            raise ValueError("collector payload requested an optimizer update")
        payload_path = Path(payload["path"]).resolve()
        if not payload_path.is_relative_to(run_dir):
            raise ValueError(f"collector payload is outside run directory: {payload_path}")
        if not payload_path.is_file():
            raise ValueError(f"collector payload is missing: {payload_path}")
        actual_hash = hashlib.sha256(payload_path.read_bytes()).hexdigest()
        if actual_hash != payload.get("sha256"):
            raise ValueError(f"collector payload hash mismatch: {payload_path}")
        payloads.append({"path": str(payload_path), "sha256": actual_hash,
                         "reward": float(row["reward"]),
                         "trajectory_ids": list(map(str, payload["trajectory_ids"]))})
    if len(payloads) < 2:
        raise ValueError("consume smoke requires at least two collected trajectories")
    return payloads


def build_smoke_advantages(payloads: list[dict]) -> dict[str, float]:
    rewards = [float(item["reward"]) for item in payloads]
    mean = sum(rewards) / len(rewards)
    variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
    std = math.sqrt(variance)
    if std > 1e-8:
        values = [(value - mean) / std for value in rewards]
    else:
        # This is a bounded integration/mutation smoke, not a learning claim: force a
        # symmetric non-zero signal so adapter mutation + checkpoint reload are exercised.
        values = [-1.0 if index % 2 == 0 else 1.0 for index in range(len(rewards))]
    advantages = {}
    for item, value in zip(payloads, values):
        for trajectory_id in item["trajectory_ids"]:
            advantages[str(trajectory_id)] = float(value)
    return advantages


def validate_update(update: dict) -> dict:
    epoch_stats = list(update.get("epoch_stats") or [])
    epoch0_ratio = epoch_stats[0].get("mean_ratio") if epoch_stats else None
    nonfinite = int(update.get("n_nonfinite", 0)) + int(update.get("post_step_n_nonfinite", 0))
    dropped = int(update.get("n_dropped", 0)) + int(update.get("post_step_n_dropped", 0))
    checks = {
        "new_update_not_replay": not bool(update.get("replayed_update", False)),
        "adapter_delta_positive": float(update.get("adapter_delta_l2", 0.0)) > 0.0,
        "no_nonfinite": nonfinite == 0,
        "no_dropped": dropped == 0,
        "epoch0_ratio_one": epoch0_ratio is not None and abs(float(epoch0_ratio) - 1.0) <= 1e-5,
    }
    return {"pass": all(checks.values()), "checks": checks,
            "epoch0_mean_ratio": epoch0_ratio, "nonfinite": nonfinite, "dropped": dropped}


class TrainerRPC:
    def __init__(self, host: str, port: int):
        self.sock = socket.create_connection((host, port), timeout=60)

    @staticmethod
    def _recv_exact(sock, size):
        data = bytearray()
        while len(data) < size:
            packet = sock.recv(size - len(data))
            if not packet:
                raise EOFError(f"trainer closed after {len(data)}/{size} bytes")
            data.extend(packet)
        return bytes(data)

    def call(self, request):
        blob = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        self.sock.sendall(struct.pack(">I", len(blob)) + blob)
        size = struct.unpack(">I", self._recv_exact(self.sock, 4))[0]
        response = pickle.loads(self._recv_exact(self.sock, size))
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(response["error"])
        return response

    def close(self):
        self.sock.close()


def consume_smoke(run_dir: Path, host: str, port: int, cfg: dict) -> dict:
    run_dir = Path(run_dir)
    audit_path = run_dir / "audit.json"
    report_path = run_dir / "consume_smoke" / "consume_smoke.json"
    if not (run_dir / "DONE").is_file() or not audit_path.is_file():
        raise RuntimeError("collection DONE and audit.json are required before consume smoke")
    if not json.loads(audit_path.read_text(encoding="utf-8")).get("pass", False):
        raise RuntimeError("collector audit did not pass")
    if report_path.exists():
        raise RuntimeError(f"consume smoke already has a terminal report: {report_path}")
    payloads = load_collector_payloads(run_dir)
    advantages = build_smoke_advantages(payloads)
    checkpoint = run_dir / "consume_smoke" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    client = TrainerRPC(host, port)
    started = time.time()
    try:
        reset = client.call({"op": "reset"})
        imports = [client.call({"op": "import_store", "path": item["path"],
                                "expected_sha256": item["sha256"]})
                   for item in payloads]
        before = client.call({"op": "metrics"})
        update = client.call({
            "op": "update", "advantages": advantages,
            "clip": float(cfg.get("clip", 0.1)),
            "kl_coef": float(cfg.get("kl_coef", 0.005)),
            "ratio_max": float(cfg.get("ratio_max", 10.0)),
            "adv_clip": float(cfg.get("adv_clip", 3.0)),
            "update_epochs": int(cfg.get("update_epochs", 2)),
            "target_kl": float(cfg.get("target_kl", 0.02)),
            "update_id": f"collector-consume-{run_dir.name}",
            "curriculum_state": None,
        })
        saved = client.call({"op": "save", "path": str(checkpoint), "update_index": 1,
                             "train_meta": {"source": "rlinf_grid_consume_smoke"}})
        loaded = client.call({"op": "load", "path": str(checkpoint)})
    finally:
        client.close()
    gate = validate_update(update)
    checkpoint_ok = checkpoint.is_file() and bool(saved) and bool(loaded)
    report = {
        "status": "PASS" if gate["pass"] and checkpoint_ok else "FAIL",
        "run_dir": str(run_dir),
        "payload_count": len(payloads),
        "trajectory_count": len(advantages),
        "advantages": advantages,
        "reset": reset,
        "imports": imports,
        "store_before_update": before,
        "update": update,
        "update_gate": gate,
        "saved": saved,
        "loaded": loaded,
        "checkpoint_reloadable": checkpoint_ok,
        "checkpoint": str(checkpoint),
        "elapsed_seconds": time.time() - started,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10088)
    parser.add_argument("--config", default="/integration/configs/grid_smoke.yaml")
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(Path(args.config).read_text()).get("consume_smoke", {})
    report = consume_smoke(Path(args.run_dir), args.host, args.port, config)
    print(json.dumps(report, indent=2, default=str))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
