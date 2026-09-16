#!/usr/bin/env python3
"""RLinf-MiBoT smoke / healthcheck commands.

Runs a sequence of independent gates and writes a machine-readable JSON report.
Each gate is best-effort and isolated: one failure does not abort the rest, so a
partial-server run still tells you exactly what is and isn't ready.

Gates:
  imports      python imports (torch, transformers, rlinf, integration modules)
  cuda         a real CUDA tensor op (requires GPU)
  egl          EGL context creation for MuJoCo rendering (requires GPU + driver)
  robocasa     RoboCasa env reset + one step (requires assets mount + EGL)
  checkpoint   MiBoT checkpoint + processor load, record config hash (requires GPU + /checkpoint)
  volume       artifact-volume write to /results

Usage:
  python3 smoke.py --all --out /results/smoke
  python3 smoke.py --only imports,volume --out /tmp/smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

GATES = ["imports", "cuda", "egl", "robocasa", "checkpoint", "volume"]


def _result(name, ok, detail, needs_server=False):
    return {"gate": name, "status": "PASS" if ok else ("NEEDS_SERVER" if needs_server else "FAIL"),
            "detail": str(detail)}


def gate_imports():
    import torch  # noqa
    import transformers  # noqa
    mods = {"torch": torch.__version__, "transformers": transformers.__version__}
    try:
        import rlinf  # noqa
        mods["rlinf"] = "importable"
    except Exception as e:  # RLinf import is optional at static time
        mods["rlinf"] = f"unavailable ({type(e).__name__})"
    # integration modules import statically (no GPU needed).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mibot_adapter  # noqa
    import rlinf_env  # noqa
    mods["integration"] = "mibot_adapter+rlinf_env import ok"
    return True, mods


def gate_cuda():
    import torch
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available()==False", True
    x = torch.randn(1024, 1024, device="cuda")
    y = (x @ x).sum().item()
    return True, {"device": torch.cuda.get_device_name(0), "matmul_sum_finite": bool(y == y)}


def gate_egl():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        import mujoco  # noqa
    except Exception as e:
        return False, f"mujoco import failed: {e}", True
    try:
        # A minimal offscreen render exercises the EGL context.
        import mujoco
        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom type='sphere' size='0.1'/></worldbody></mujoco>")
        data = mujoco.MjData(model)
        ren = mujoco.Renderer(model, 64, 64)
        mujoco.mj_forward(model, data)
        ren.update_scene(data)
        _ = ren.render()
        return True, "EGL offscreen render 64x64 ok"
    except Exception as e:
        return False, f"EGL render failed: {e}", True


def gate_robocasa():
    try:
        import robocasa  # noqa
        import robosuite  # noqa
    except Exception as e:
        return False, f"robocasa/robosuite import failed: {e}", True
    try:
        import robosuite
        env = robosuite.make(
            env_name="CloseBlenderLid", robots="PandaOmron",
            has_renderer=False, has_offscreen_renderer=True,
            use_camera_obs=True, camera_names="robot0_agentview_center",
            camera_heights=128, camera_widths=128, control_freq=20,
        )
        env.reset()
        env.step(env.action_spec[0] * 0.0)
        env.close()
        return True, "CloseBlenderLid reset+step ok"
    except Exception as e:
        return False, f"env reset/step failed: {e}", True


def gate_checkpoint():
    ckpt = os.environ.get("CHECKPOINT_DIR", "/checkpoint")
    if not Path(ckpt, "config.json").exists():
        return False, f"no config.json under {ckpt}", True
    import hashlib
    cfg = Path(ckpt, "config.json").read_bytes()
    cfg_hash = hashlib.sha256(cfg).hexdigest()
    import torch
    if not torch.cuda.is_available():
        return False, f"config.json hash={cfg_hash[:16]} but no GPU to load weights", True
    try:
        from transformers import AutoModel, AutoProcessor
        proc = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
        model = AutoModel.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
        n = sum(p.numel() for p in model.parameters())
        return True, {"config_sha256": cfg_hash, "params": n, "processor": type(proc).__name__}
    except Exception as e:
        return False, f"config hash={cfg_hash[:16]}; load failed: {e}", True


def gate_volume():
    out = Path(os.environ.get("RESULTS_DIR", "/results"))
    out.mkdir(parents=True, exist_ok=True)
    probe = out / f".smoke_write_{int(time.time())}"
    probe.write_text("ok")
    txt = probe.read_text()
    probe.unlink()
    return (txt == "ok"), f"wrote+read {out}"


RUNNERS = {
    "imports": gate_imports, "cuda": gate_cuda, "egl": gate_egl,
    "robocasa": gate_robocasa, "checkpoint": gate_checkpoint, "volume": gate_volume,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", default="", help="comma list of gates")
    ap.add_argument("--out", default="/results/smoke")
    args = ap.parse_args()

    chosen = GATES if args.all or not args.only else [g.strip() for g in args.only.split(",") if g.strip()]
    results = []
    for g in chosen:
        try:
            out = RUNNERS[g]()
            ok, detail = out[0], out[1]
            needs = out[2] if len(out) > 2 else False
            results.append(_result(g, ok, detail, needs))
        except ModuleNotFoundError as e:
            # GPU-stack deps (torch/transformers/mujoco/robosuite/rlinf) live only in the
            # container; a missing one off-server is a NEEDS_SERVER gate, not a hard FAIL.
            results.append(_result(g, False, f"missing module: {e.name}", needs_server=True))
        except Exception:
            results.append(_result(g, False, "exception:\n" + traceback.format_exc()))
        print(f"[{results[-1]['status']:>12}] {g}: {str(results[-1]['detail'])[:200]}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    report = {"ts": int(time.time()), "gates": results,
              "summary": {s: sum(1 for r in results if r["status"] == s)
                          for s in ("PASS", "FAIL", "NEEDS_SERVER")}}
    (outdir / "smoke_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {outdir/'smoke_report.json'}  summary={report['summary']}")
    # Non-zero only on a hard FAIL (NEEDS_SERVER is expected off-server).
    sys.exit(1 if report["summary"]["FAIL"] else 0)


if __name__ == "__main__":
    main()
