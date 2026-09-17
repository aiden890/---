from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from checkpoint_schema import (  # noqa: E402
    build_checkpoint_metadata, rng_state_for_restore, validate_checkpoint_metadata,
)


def valid():
    return build_checkpoint_metadata(
        adapter_skills=["grasp"], targets=["dit.layers.0.attn.qkv_proj"],
        rank=8, alpha=32, base_model="Xiaomi-Robotics-1-RoboCasa365",
        sampler="pirl", config={"eta": 0.1}, source_commit="abc123",
        source_manifest_sha256="deadbeef", update_index=2,
    )


def test_metadata_roundtrip():
    meta = valid()
    assert meta["source_manifest_sha256"] == "deadbeef"
    validate_checkpoint_metadata(meta, adapter_skills=["grasp"],
                                 targets=["dit.layers.0.attn.qkv_proj"], rank=8, alpha=32)


def test_metadata_rejects_target_order_change_missing_shape_and_legacy():
    cases = []
    m = valid(); m["targets"] = list(reversed(m["targets"])) + ["extra"]; cases.append(m)
    m = valid(); m["adapter_skills"] = ["place"]; cases.append(m)
    m = valid(); m["rank"] = 4; cases.append(m)
    m = valid(); m["schema_version"] = 1; cases.append(m)
    m = valid(); del m["source_manifest_sha256"]; cases.append(m)
    for mutated in cases:
        try:
            validate_checkpoint_metadata(mutated, adapter_skills=["grasp"],
                                         targets=["dit.layers.0.attn.qkv_proj"], rank=8, alpha=32)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"checkpoint metadata mutation escaped: {mutated}")


def test_rng_restore_normalizes_all_tensor_states_to_cpu():
    class FakeTensor:
        def __init__(self, name):
            self.name = name
            self.cpu_calls = 0

        def cpu(self):
            self.cpu_calls += 1
            return f"cpu:{self.name}"

    torch_state = FakeTensor("torch")
    cuda_state = FakeTensor("cuda:0")
    original = {
        "torch": torch_state,
        "cuda": [cuda_state],
        "python": (3, (1, 2), None),
    }
    normalized = rng_state_for_restore(original)
    assert normalized["torch"] == "cpu:torch"
    assert normalized["cuda"] == ["cpu:cuda:0"]
    assert torch_state.cpu_calls == cuda_state.cpu_calls == 1
    assert normalized["python"] is original["python"]
    assert normalized is not original


if __name__ == "__main__":
    test_metadata_roundtrip()
    test_metadata_rejects_target_order_change_missing_shape_and_legacy()
    test_rng_restore_normalizes_all_tensor_states_to_cpu()
    print("checkpoint schema tests passed")
