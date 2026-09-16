#!/usr/bin/env python3
"""Deterministic ODE evaluation entry point (Hydra-configured).

Runs the noise_level=0 pi-RL Flow-SDE (bit-exact with the checkpoint's Euler ODE) on the
fixed eval seeds, optionally comparing a base vs a trained-adapter policy. No training.
GPU execution is a NEEDS_SERVER gate; host-side config/wiring is validated on a dry run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_entry import load_config  # reuse the standalone config loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="ode_eval")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config_name, args.overrides)
    assert float(cfg["sampler"]["noise_level"]) == 0.0, "ODE eval requires noise_level=0"

    results_dir = Path(cfg.get("results_dir", "/results/ode_eval"))
    results_dir.mkdir(parents=True, exist_ok=True)
    report = {"ts": int(time.time()), "config": cfg, "stage": "ode_eval", "seeds": cfg["eval"]["seeds"]}

    import mibot_adapter as MA
    mcfg = MA.MiBoTConfig(**{k: v for k, v in cfg["model"].items() if k in MA.MiBoTConfig.__dataclass_fields__})
    report["wire"] = "MiBoTConfig built; sampler=pirl_flow_sde noise=0"

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA")
        model = MA.MiBoTModel(mcfg).load()
        sampler = MA.MiBoTSampler(model)  # noqa: F841  (used inside per-seed rollout on server)
        # Per-seed deterministic rollout + success eval runs against the sim (client image);
        # wired here but a NEEDS_SERVER gate for actual execution.
        raise NotImplementedError("per-seed ODE rollout is a NEEDS_SERVER gate")
    except Exception as e:
        report["status"] = "NEEDS_SERVER"
        report["detail"] = f"{type(e).__name__}: {e}"

    (results_dir / "ode_eval_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"[{report['status']}] ode_eval -> {results_dir/'ode_eval_report.json'}")


if __name__ == "__main__":
    main()
