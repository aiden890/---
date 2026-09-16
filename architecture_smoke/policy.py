"""Base-policy interface for the pinned Xiaomi RoboCasa365 VLA.

Adapter-free contract for this smoke test:
  * adapter_mode is asserted to be disabled/base_only,
  * adapter_checkpoint MUST be None -- construction raises otherwise,
  * normalization + deterministic sampler are the checkpoint defaults (rollout.py
    EvalClient is used verbatim: replan 16, obs 4/2, crop 0.95).

The heavy imports (numpy, rollout.EvalClient) are lazy so schemas/planner/
verifier/executor can be unit-tested locally without a GPU, numpy, or the
checkpoint. A ``MockPolicy`` (plain-list actions) with the same interface drives
the offline end-to-end control-loop check.
"""
from __future__ import annotations

from schemas import AdapterMode, PolicyInput, PolicyOutput


def _assert_adapter_disabled(pin: PolicyInput) -> None:
    if pin.adapter_mode not in (AdapterMode.DISABLED, AdapterMode.BASE_ONLY):
        raise AssertionError(f"adapter_mode must be disabled/base_only, got {pin.adapter_mode}")
    if pin.adapter_checkpoint is not None:
        raise AssertionError(
            f"adapter-free smoke test: no adapter checkpoint may be loaded, "
            f"got {pin.adapter_checkpoint!r}"
        )


class BasePolicyClient:
    """Wraps rollout.EvalClient. Base checkpoint only, no adapter."""

    def __init__(self, model_path, server_addr, server_port, robot_type, crop_ratio,
                 replan_steps, adapter_mode=AdapterMode.DISABLED):
        if adapter_mode not in (AdapterMode.DISABLED, AdapterMode.BASE_ONLY):
            raise AssertionError(f"BasePolicyClient refuses adapter_mode={adapter_mode}")
        self.adapter_mode = adapter_mode
        self.adapter_checkpoint = None          # invariant: never load an adapter
        self.replan_steps = replan_steps
        self.model_path = model_path
        import numpy as np                       # lazy
        import rollout                            # lazy: pulls torch / client / processor
        self._np = np
        self._rollout = rollout
        self._client = rollout.EvalClient(model_path, server_addr, server_port,
                                           robot_type, crop_ratio)

    def infer(self, pin: PolicyInput) -> PolicyOutput:
        _assert_adapter_disabled(pin)
        np = self._np
        chunk = self._client.infer(pin.state_history, pin.image_history, pin.instruction)
        chunk = np.asarray(chunk, dtype=np.float32)
        if len(chunk) < self.replan_steps:
            raise RuntimeError(f"policy returned {len(chunk)} actions < replan {self.replan_steps}")
        return PolicyOutput(action_chunk=chunk[: self.replan_steps], chunk_len=int(len(chunk)),
                            adapter_mode=self.adapter_mode, adapter_checkpoint=None)

    def provenance(self) -> dict:
        return {"model_path": self.model_path, "adapter_mode": self.adapter_mode.value,
                "adapter_checkpoint": self.adapter_checkpoint, "replan_steps": self.replan_steps}

    def close(self):
        self._client.close()


class MockPolicy:
    """Interface-compatible policy for the offline control-loop check.

    Emits a deterministic plain-list zero action chunk (no numpy needed); still
    enforces the adapter-disabled assertion so the check exercises the guard.
    """

    def __init__(self, replan_steps=16, action_dim=12, adapter_mode=AdapterMode.DISABLED):
        self.replan_steps = replan_steps
        self.action_dim = action_dim
        self.adapter_mode = adapter_mode
        self.adapter_checkpoint = None
        self.calls = 0

    def infer(self, pin: PolicyInput) -> PolicyOutput:
        _assert_adapter_disabled(pin)
        self.calls += 1
        chunk = [[0.0] * self.action_dim for _ in range(self.replan_steps)]
        return PolicyOutput(action_chunk=chunk, chunk_len=self.replan_steps,
                            adapter_mode=self.adapter_mode, adapter_checkpoint=None)

    def provenance(self) -> dict:
        return {"model_path": "MOCK", "adapter_mode": self.adapter_mode.value,
                "adapter_checkpoint": None, "replan_steps": self.replan_steps}

    def close(self):
        pass
