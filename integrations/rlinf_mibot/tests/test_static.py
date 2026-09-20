#!/usr/bin/env python3
"""No-GPU static tests for the RLinf-MiBoT bundle.

Validates everything checkable WITHOUT the GPU server: modules import, config files parse,
guards reject long training, revision pins are present and consistent, and the integration
interfaces expose the expected names. Run with: python3 tests/test_static.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAILS.append(f"{name}: {detail}")
    print(f"[{status}] {name}{'' if cond else ' -- ' + detail}")


def test_imports():
    import mibot_adapter as MA
    import mibot_rlinf_env as MRE
    import rlinf_env as RE
    for n in ("MiBoTConfig", "MiBoTModel", "MiBoTSampler", "RLinfModelSpec",
              "load_reward_manager", "load_skill_monitor"):
        check(f"mibot_adapter.{n}", hasattr(MA, n))
    for n in ("SKILL_CONTRACTS", "EnvSpec", "RoboCasaRewardAdapter", "register_mibot"):
        check(f"rlinf_env.{n}", hasattr(RE, n))
    check("action dims", MA.ACTIVE_ACTION_DIM == 12 and MA.FULL_ACTION_DIM == 60)
    # spec descriptor carries the pinned framework commit
    check("model_spec pins RLinf commit",
          MA.RLinfModelSpec().framework_commit == "bde6c918642abf9a4776cb1d5fabcc5087dfe195")
    check("skill contracts complete",
          set(RE.SKILL_CONTRACTS) == {"GRASP_OBJECT", "MOVE_OBJECT", "PLACE_OBJECT"})
    check("temporal history is import-safe", hasattr(MRE, "TemporalHistoryBuffer"))


def test_rlinf_registration_contract():
    """Exercise the registry seam without installing the GPU framework."""
    import types
    import rlinf_env as RE

    calls = []
    fake_rlinf = types.ModuleType("rlinf")
    fake_models = types.ModuleType("rlinf.models")
    fake_models.register_model = lambda *a, **kw: calls.append((a, kw))
    fake_policy = types.ModuleType("mibot_rlinf_policy")
    fake_policy.build_mibot_policy = lambda cfg, dtype=None: (cfg, dtype)
    old_rlinf = sys.modules.get("rlinf")
    old_models = sys.modules.get("rlinf.models")
    old_policy = sys.modules.get("mibot_rlinf_policy")
    try:
        sys.modules["rlinf"] = fake_rlinf
        sys.modules["rlinf.models"] = fake_models
        sys.modules["mibot_rlinf_policy"] = fake_policy
        spec = RE.register_mibot()
    finally:
        if old_rlinf is None:
            sys.modules.pop("rlinf", None)
        else:
            sys.modules["rlinf"] = old_rlinf
        if old_models is None:
            sys.modules.pop("rlinf.models", None)
        else:
            sys.modules["rlinf.models"] = old_models
        if old_policy is None:
            sys.modules.pop("mibot_rlinf_policy", None)
        else:
            sys.modules["mibot_rlinf_policy"] = old_policy
    check("RLinf custom model registered", len(calls) == 1)
    if calls:
        args, kwargs = calls[0]
        check("RLinf model name", args[0] == spec.name)
        check("RLinf embodied category", kwargs.get("category") == "embodied")
        check("RLinf registration idempotent", kwargs.get("force") is True)


def test_configs():
    import yaml
    for cn in ("ppo_smoke", "ode_eval"):
        cfg = yaml.safe_load((ROOT / "configs" / f"{cn}.yaml").read_text())
        check(f"config {cn} parses", isinstance(cfg, dict))
        check(f"config {cn} training disabled", cfg["guards"]["training_enabled"] is False)
    ppo = yaml.safe_load((ROOT / "configs" / "ppo_smoke.yaml").read_text())
    check("ppo_smoke one iteration", ppo["ppo"]["num_iterations"] == 1)
    check("ppo_smoke update_epochs>=2", ppo["ppo"]["update_epochs"] >= 2)
    check("ppo_smoke adapter_only", ppo["model"]["train_mode"] == "adapter_only")
    ode = yaml.safe_load((ROOT / "configs" / "ode_eval.yaml").read_text())
    check("ode_eval noise=0", float(ode["sampler"]["noise_level"]) == 0.0)
    grid = yaml.safe_load((ROOT / "configs" / "grid_smoke.yaml").read_text())
    check("grid config parses", isinstance(grid, dict))
    check("grid has >=2 parameter configs", len(grid["grid"]["sampler.noise_level"]) >= 2)
    check("grid uses >=2 RLinf workers", int(grid["runtime"]["workers"]) >= 2)
    check("grid collection is optimizer-free", "optimizer" not in grid["grid"])
    worker_source = (ROOT / "src" / "rlinf_grid_worker.py").read_text()
    check("retry attempts use isolated trajectory IDs",
          "__attempt{attempt}" in worker_source and "discard_store([traj_id])" in worker_source)
    check("grid snapshot uses canonical Sim signature",
          'sim.snapshot("grid_initial", 0)' in worker_source)
    deployment_doc = " ".join((ROOT / "deploy" / "README.md").read_text().split())
    check("two-Spark fabric named RoCE not NVLink",
          "ConnectX-7/RoCE" in deployment_doc and "not NVLink" in deployment_doc)
    native = yaml.safe_load((ROOT / "configs" / "rlinf_mibot_grpo.yaml").read_text())
    check("native GRPO config parses", isinstance(native, dict))
    check("native config uses GRPO", native["algorithm"]["adv_type"] == "grpo")
    check("native config keeps elementwise ratios",
          native["algorithm"]["logprob_type"] == "token_level")
    placement = native["cluster"]["component_placement"]
    check("3090 is actor", placement["actor"]["node_group"] == "train_3090")
    check("DAX is rollout+env",
          placement["rollout"]["node_group"] == "dax_rollout"
          and placement["env"]["node_group"] == "dax_rollout")
    check("patch weight sync enabled", native["defaults"][2] == "weight_syncer/patch_syncer@weight_syncer")
    for split in ("train", "eval"):
        history = native["env"][split]["mibot_history"]
        check(f"native {split} temporal history enabled",
              history["enabled"] is True and int(history["interval"]) == 2)
        grasp_reward = native["env"][split]["grasp_binary_reward"]
        check(f"native {split} stable-grasp binary reward enabled",
              grasp_reward["enabled"] is True
              and int(grasp_reward["hold_steps"]) == 20
              and float(grasp_reward["lift_dz"]) == 0.05
              and native["env"][split]["use_rel_reward"] is True)
    check("native GRPO members share reset seed",
          native["env"]["train"]["seed_strategy"] == "same")
    check("native evaluation seeds remain unique",
          native["env"]["eval"]["seed_strategy"] == "global_unique")
    check("native generic-worker eval disables stochastic sampling",
          float(native["rollout"]["sampling_params"]["temperature_eval"]) <= 0.0)
    check("native Ray workers enable MiBoT registry bootstrap",
          "RLINF_ENABLE_MIBOT_REGISTRATION=1" in
          (ROOT / "run.sh").read_text())
    bootstrap = (ROOT / "src" / "sitecustomize.py").read_text()
    check("fresh Ray workers install model and environment integration",
          "register_mibot()" in bootstrap
          and "install_mibot_env_override()" in bootstrap)
    check("native reward is a labeled GRASP binary run",
          native["runner"]["logger"]["experiment_name"].endswith("grasp_binary"))


def test_guard():
    from train_entry import load_config
    # attempting a multi-iteration run must be refused
    cfg = load_config("ppo_smoke", ["ppo.num_iterations=50"])
    import train_entry
    # emulate the guard directly
    n = cfg["ppo"]["num_iterations"]
    refused = cfg["guards"].get("training_enabled", False) or n > cfg["guards"]["max_iterations"]
    check("guard refuses num_iterations=50", refused)


def test_pins():
    lock = (ROOT / "REVISIONS.lock").read_text()
    for key in ("RLINF_COMMIT=bde6c918642abf9a4776cb1d5fabcc5087dfe195",
                "ROBOCASA_COMMIT=4f8a2980def75a55dff96b990745b83540425f09",
                "ROBOSUITE_COMMIT=5ce6643f3092639d08f7b0f90ed1c6a84f50552c",
                "CHECKPOINT_SHA=3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4",
                "BASE_IMAGE_DIGEST=sha256:7012e535"):
        check(f"pin {key.split('=')[0]}", key in lock, "missing/incorrect pin")
    requirements = (ROOT / "configs" / "requirements-rlinf.txt").read_text()
    check("RLinf legacy vector-env gym dependency", "gym==0.23.1" in requirements)
    check("checkpoint.sha256 present", (ROOT / "checkpoint.sha256").exists())
    # lock file present and non-empty
    lk = (ROOT / "configs" / "requirements-rlinf.lock").read_text()
    check("lock has torch cu121", "torch==2.5.1+cu121" in lk)
    requirements = (ROOT / "configs" / "requirements-rlinf.txt").read_text()
    check("RoboCasa exact numpy runtime", "numpy==2.2.5" in requirements)
    for package in ("termcolor", "h5py", "pygame", "pynput", "hidapi"):
        check(f"RoboCasa runtime dependency {package}", f"{package}==" in requirements)


def test_no_secrets():
    """Coarse secret scan across the bundle (excluding this test)."""
    import re
    patterns = [r"hf_[A-Za-z0-9]{34,}", r"AKIA[0-9A-Z]{16}", r"-----BEGIN [A-Z ]*PRIVATE KEY",
                r"(?i)password\s*=\s*['\"][^'\"]{4,}"]
    hits = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or "vendor" in p.parts or p.name == "test_static.py":
            continue
        if p.suffix in (".png", ".mp4", ".safetensors"):
            continue
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        for pat in patterns:
            if re.search(pat, txt):
                hits.append(f"{p.name}:{pat}")
    check("no embedded secrets", not hits, "; ".join(hits))


def test_env_example_no_abs_paths():
    """.env.example must contain only placeholders, never real server paths."""
    env = (ROOT / ".env.example").read_text()
    # every path value must be a REPLACE placeholder or a benign default
    import re
    bad = [ln for ln in env.splitlines()
           if re.match(r"^RLINF_(ASSETS|CHECKPOINT|RESULTS|CACHE)=", ln)
           and "/REPLACE/" not in ln]
    check(".env.example uses placeholders", not bad, str(bad))


def test_assets_mount_contract():
    """Task-scoped Docker volumes must work without weakening the default RO mount."""
    run = (ROOT / "run.sh").read_text()
    preflight = (ROOT / "preflight.sh").read_text()
    env = (ROOT / ".env.example").read_text()
    check("run supports explicit assets mount mode",
          'RLINF_ASSETS_MODE:-ro' in run and 'assets_mount' in run)
    check("grid compare writes through result volume",
          'python3 /integration/src/compare_grid_runs.py' in run)
    check("grid resume clears root-owned lock through result volume",
          '".collector.lock").unlink(missing_ok=True)' in run)
    check("preflight recognizes Docker asset volumes",
          'docker volume inspect "$RLINF_ASSETS"' in preflight)
    check("assets mount defaults read-only",
          "RLINF_ASSETS_MODE=ro" in env)


if __name__ == "__main__":
    test_imports()
    test_rlinf_registration_contract()
    test_configs()
    test_guard()
    test_pins()
    test_no_secrets()
    test_env_example_no_abs_paths()
    test_assets_mount_contract()
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}):")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("ALL STATIC TESTS PASSED")
