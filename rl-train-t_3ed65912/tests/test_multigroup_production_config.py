from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "multigroup_production.py"
CONFIG = ROOT / "configs" / "multigroup_grasp_v1.json"
ARM_B_CONFIG = ROOT / "configs" / "multigroup_grasp_arm_b_lr1e5.json"
sys.path.insert(0, str(ROOT / "scripts"))
from multigroup_production import audit_nonjoint_source  # noqa: E402


def test_production_config_dry_run_is_safe_and_complete():
    with tempfile.TemporaryDirectory() as td:
        output = Path(td) / "plan.json"
        subprocess.run([
            sys.executable, str(SCRIPT), "--config", str(CONFIG),
            "--run", "multigroup_gate_test", "--dry-run", "--out", str(output),
        ], check=True)
        plan = json.loads(output.read_text())
    assert plan["validation"] == "pass"
    assert plan["gpu_gate"] == "pending GPU gate"
    assert plan["run"] == "multigroup_gate_test"
    assert plan["client"]["group"] == 8
    assert plan["client"]["groups_per_update"] >= 8
    assert plan["client"]["iters"] == 2
    assert plan["client"]["update_epochs"] >= 2
    assert plan["client"]["sampler"] == "pirl"
    assert plan["client"]["joint_logprob"] is False
    assert plan["source_audit"] == {
        "new_logprob_elementwise_masked": True,
        "old_logprob_elementwise_masked": True,
        "ratio_exp_is_elementwise": True,
    }
    assert plan["expected"]["trajectories_per_update"] >= 64
    assert plan["expected"]["logical_update_rpcs"] == 2
    assert plan["expected"]["optimizer_epochs_per_update"] == 2
    assert "scripts/deploy.sh" not in plan["commands"]["dry_run"]
    assert "docker" not in plan["commands"]["dry_run"]
    assert "trainer-start" in plan["commands"]["trainer"]
    assert "--sampler pirl" in plan["commands"]["trainer"]
    assert "--lora-targets all_linear" in plan["commands"]["trainer"]
    assert "train multigroup_gate_test" in plan["commands"]["client"]
    assert "--groups-per-update 8" in plan["commands"]["client"]
    assert "--adaptive-band 1 7" in plan["commands"]["client"]


def test_nonjoint_source_audit_catches_joint_ratio_and_mask_mutations():
    source = (ROOT / "src" / "grpo_trainer_server.py").read_text()
    assert all(audit_nonjoint_source(source).values())
    mutations = [
        source.replace("return per[exec_mask]", "return per[exec_mask].sum().reshape(1)"),
        source.replace("return old[mask].to(", "return old[mask].sum().reshape(1).to("),
        source.replace("ratio = torch.exp(valid_lr)", "ratio = torch.exp(valid_lr.sum())"),
    ]
    for mutated in mutations:
        try:
            audit_nonjoint_source(mutated)
        except ValueError:
            pass
        else:
            raise AssertionError("joint-ratio source mutation escaped static gate")


def test_canonical_runner_exposes_cpu_only_dry_run():
    completed = subprocess.run([
        "bash", str(ROOT / "scripts" / "run-train.sh"),
        "multigroup-dry-run", "runner_gate_test",
    ], check=True, text=True, capture_output=True)
    plan = json.loads(completed.stdout)
    assert plan["validation"] == "pass"
    assert plan["gpu_gate"] == "pending GPU gate"
    assert plan["run"] == "runner_gate_test"


def test_gpu_gate_refuses_a_preexisting_trainer():
    source = (ROOT / "scripts" / "run-train.sh").read_text()
    gate = source.split("multigroup-gpu-gate)", 1)[1].split(";;", 1)[0]
    assert "trainer already exists; refusing canonical gate" in gate
    assert "docker ps" in gate


def test_arm_b_cap_covers_measured_chunk_floor_shortfall():
    config = json.loads(ARM_B_CONFIG.read_text())
    client = config["client"]
    assert client["max_groups_per_update"] >= 32
    assert client["adaptive_universe"] >= client["max_groups_per_update"]


def test_execute_failure_writes_failed_sentinel():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        runner = root / "fail-runner.sh"
        runner.write_text("#!/usr/bin/env bash\nexit 23\n")
        runner.chmod(0o755)
        completed = subprocess.run([
            sys.executable, str(SCRIPT), "--config", str(ARM_B_CONFIG),
            "--run", "sentinel_failure_test", "--execute",
            "--runner", str(runner), "--results-root", str(root / "results"),
        ], text=True, capture_output=True)
        run_dir = root / "results" / "sentinel_failure_test"
        assert completed.returncode != 0
        assert (run_dir / "FAILED").is_file()
        assert not (run_dir / "DONE").exists()
        failure = json.loads((run_dir / "FAILED").read_text())
        assert failure["status"] == "failed"
        assert "returned non-zero exit status 23" in failure["error"].lower()


if __name__ == "__main__":
    test_production_config_dry_run_is_safe_and_complete()
    test_nonjoint_source_audit_catches_joint_ratio_and_mask_mutations()
    test_canonical_runner_exposes_cpu_only_dry_run()
    test_gpu_gate_refuses_a_preexisting_trainer()
    test_arm_b_cap_covers_measured_chunk_floor_shortfall()
    test_execute_failure_writes_failed_sentinel()
    print("6 production dry-run/source-mutation tests passed")
