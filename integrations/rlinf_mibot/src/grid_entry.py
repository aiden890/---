#!/usr/bin/env python3
"""Hydra/OmegaConf entry point for grid collect/audit/resume/consume operations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def partition_grid_overrides(grid_keys: set[str], overrides: list[str]):
    grid, regular = [], []
    for item in overrides:
        key = item.split("=", 1)[0]
        (grid if key in grid_keys else regular).append(item)
    return grid, regular


def load_config(path: Path, overrides: list[str]) -> dict:
    from omegaconf import OmegaConf
    config = OmegaConf.load(path)
    grid_overrides, regular_overrides = partition_grid_overrides(set(config.grid), overrides)
    if regular_overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(regular_overrides))
    for item in grid_overrides:
        key, value = item.split("=", 1)
        config.grid[key] = OmegaConf.from_dotlist([f"value={value}"]).value
    return OmegaConf.to_container(config, resolve=True)


def resolve_provenance(cfg: dict) -> dict:
    cfg = json.loads(json.dumps(cfg))
    if cfg.get("run", {}).get("source_commit") == "auto":
        source_commit = os.environ.get("RLINF_SOURCE_COMMIT")
        if not source_commit:
            try:
                source_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True,
                    stderr=subprocess.DEVNULL).strip()
            except subprocess.CalledProcessError as exc:
                raise RuntimeError("RLINF_SOURCE_COMMIT is required outside a git checkout") from exc
        cfg["run"]["source_commit"] = source_commit
    checkpoint_hash = cfg.get("run", {}).get("checkpoint_hash")
    if checkpoint_hash == "from-preflight":
        checksum = Path(__file__).resolve().parents[1] / "checkpoint.sha256"
        cfg["run"]["checkpoint_hash"] = hashlib.sha256(checksum.read_bytes()).hexdigest()
    return cfg


def output_path(cfg: dict, mode: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    run_id = str(cfg["run"]["id"])
    if mode == "serial":
        run_id += "-serial"
    return Path(cfg.get("results_root", "/results")) / run_id


def parse_cli(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("collect", "resume", "audit", "fake-smoke", "consume-smoke"))
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "grid_smoke.yaml"))
    parser.add_argument("--output")
    parser.add_argument("--mode", choices=("parallel", "serial"), default="parallel")
    parser.add_argument("overrides", nargs="*")
    # Launcher options precede Hydra-style positional overrides. parse_args() rejects this
    # supported intermixing when the final positional uses nargs="*".
    return parser.parse_intermixed_args(argv)


def main():
    args = parse_cli()

    overrides = [item for item in args.overrides if item != "--"]
    cfg = resolve_provenance(load_config(Path(args.config), overrides))
    if args.mode == "serial":
        cfg["runtime"]["workers"] = 1
        cfg["runtime"]["max_in_flight"] = 1
        cfg["runtime"]["node_ranks"] = [0]
    out = output_path(cfg, args.mode, args.output)

    if args.command in {"collect", "resume"}:
        from rlinf_grid_runtime import collect_with_rlinf
        result = collect_with_rlinf(cfg, out)
    elif args.command == "fake-smoke":
        from grid_collection import FakeEpisodeBackend, collect_grid
        result = collect_grid(cfg, out, FakeEpisodeBackend(delay_seconds=0.05))
    elif args.command == "audit":
        from grid_collection import audit_collection
        result = audit_collection(out)
    else:
        from consume_collector import consume_smoke
        server = cfg["model_server"]
        result = consume_smoke(out, server["host"], int(server["port"]), cfg["consume_smoke"])
    print(json.dumps(result, indent=2, default=str))
    if result.get("status") == "failed" or result.get("pass") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
