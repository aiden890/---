from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from smoke_gate import evaluate_smoke  # noqa: E402


def good_row(i=0):
    return {
        "iter": i,
        "trajectories_collected": 64,
        "groups_collected": 8,
        "trainable_chunks_collected": 1024,
        "group_env_seeds": list(range(i * 8, i * 8 + 8)),
        "epoch0_mean_ratio": 1.0,
        "epoch_stats": [{"epoch": 0, "mean_abs_logratio": 0.0, "n_nonfinite": 0}],
        "post_step_n_nonfinite": 0,
        "adapter_delta_l2": 0.01,
        "peak_mem_gb": 18.0,
        "grad_norm": 2.0,
        "loss": 0.2,
    }


def test_clean_two_update_smoke_passes():
    report = evaluate_smoke([good_row(0), good_row(1)])
    assert report["pass"]
    assert report["checks"]["at_least_128_trajectories"]


def test_gate_mutations_are_caught():
    mutations = {
        "chunks": lambda rows: rows[0].update(trainable_chunks_collected=1023),
        "ratio": lambda rows: rows[0].update(epoch0_mean_ratio=1.01),
        "logprob": lambda rows: rows[0]["epoch_stats"][0].update(mean_abs_logratio=0.01),
        "nonfinite": lambda rows: rows[0].update(post_step_n_nonfinite=1),
        "delta": lambda rows: rows[0].update(adapter_delta_l2=0.0),
        "duplicate_seed": lambda rows: rows[0].update(group_env_seeds=[0] * 8),
        "trajectory_count": lambda rows: rows[0].update(trajectories_collected=63),
    }
    for name, mutate in mutations.items():
        rows = [good_row(0), good_row(1)]
        mutate(rows)
        report = evaluate_smoke(copy.deepcopy(rows))
        assert not report["pass"], f"mutation escaped gate: {name}"


if __name__ == "__main__":
    test_clean_two_update_smoke_passes()
    test_gate_mutations_are_caught()
