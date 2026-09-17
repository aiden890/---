from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from checkpoint_schema import build_checkpoint_metadata, validate_checkpoint_metadata  # noqa: E402


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


if __name__ == "__main__":
    test_metadata_roundtrip()
    test_metadata_rejects_target_order_change_missing_shape_and_legacy()
    print("checkpoint schema tests passed")
