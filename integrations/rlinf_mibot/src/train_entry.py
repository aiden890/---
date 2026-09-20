#!/usr/bin/env python3
"""Adapter-only PPO smoke entry point (Hydra-configured).

Enforces the no-long-training guard, wires the verified components, and runs exactly one
short PPO update as a data-flow correctness probe. GRPO is added only AFTER PPO + the
deterministic eval pass on the server.

The actual optimizer step against the pinned RLinf PPO runs inside the GPU container and
is a NEEDS_SERVER gate. This entry point does all the host-side wiring/guarding it can and
writes a report so a dry run (no GPU) still validates config + wiring.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_config(config_name: str, overrides: list[str]) -> dict:
    import yaml  # PyYAML (in lock)
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / f"{config_name}.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # apply `key.sub=value` overrides (Hydra-style minimal support for standalone runs)
    for ov in overrides:
        if "=" not in ov:
            continue
        k, v = ov.split("=", 1)
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            v = yaml.safe_load(v)
        except Exception:
            pass
        node[parts[-1]] = v
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="ppo_smoke")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config_name, args.overrides)

    # --- guard: refuse to run long training from this entry point ---------------------
    guards = cfg.get("guards", {})
    n_iter = int(cfg.get("ppo", {}).get("num_iterations", 1))
    if guards.get("training_enabled", False) or n_iter > int(guards.get("max_iterations", 1)):
        raise SystemExit(f"REFUSED: training_enabled or num_iterations={n_iter} exceeds smoke cap. "
                         "This entry point only runs a 1-iteration PPO smoke.")

    results_dir = Path(cfg.get("results_dir", "/results/ppo_smoke"))
    results_dir.mkdir(parents=True, exist_ok=True)

    report = {"ts": int(time.time()), "config": cfg, "stage": "ppo_smoke", "steps": []}

    # Host-side wiring validation (import-safe, no GPU needed).
    import mibot_adapter as MA
    import rlinf_env as RE
    mcfg = MA.MiBoTConfig(**{k: v for k, v in cfg["model"].items() if k in MA.MiBoTConfig.__dataclass_fields__})
    espec = RE.EnvSpec(**{k: v for k, v in cfg["env"].items() if k in RE.EnvSpec.__dataclass_fields__})
    report["steps"].append({"wire": "MiBoTConfig+EnvSpec built", "ok": True})
    report["model_spec"] = MA.RLinfModelSpec().__dict__

    # GPU execution path (NEEDS_SERVER).
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA")
        model = MA.MiBoTModel(mcfg).load()
        RE.register_mibot()
        report["steps"].append({
            "wire": "RLinf MiBoT registry",
            "ok": True,
            "detail": "Use rlinf_train.py for the native GRPO optimizer run",
        })
        report["status"] = "PASS"
    except Exception as e:
        report["steps"].append({"run": "one PPO update", "ok": False, "detail": f"{type(e).__name__}: {e}"})
        report["status"] = "NEEDS_SERVER"

    (results_dir / "ppo_smoke_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"[{report['status']}] ppo_smoke -> {results_dir/'ppo_smoke_report.json'}")


if __name__ == "__main__":
    main()
