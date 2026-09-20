from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import grpo_trainer_server as subject  # noqa: E402


def test_rollout_inputs_store_only_model_visible_precision():
    server = subject.GRPOTrainerServer.__new__(subject.GRPOTrainerServer)
    server.model = SimpleNamespace(device=torch.device("cpu"), dtype=torch.bfloat16)
    inputs = {
        "pixel_values_videos": torch.tensor([[1.0001, -2.0002]], dtype=torch.float32),
        "state": torch.tensor([0.125], dtype=torch.float32),
        "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
        "instruction": "grasp",
    }

    stored = server._store_inputs_cpu(inputs)

    assert stored["pixel_values_videos"].dtype == torch.bfloat16
    assert stored["state"].dtype == torch.bfloat16
    assert stored["input_ids"].dtype == torch.int64
    assert stored["instruction"] == "grasp"
    for key in ("pixel_values_videos", "state", "input_ids"):
        # Recompute sees exactly what the rollout forward saw through _to_dev.
        assert torch.equal(server._to_dev(inputs[key]), server._to_dev(stored[key]))
    assert (stored["pixel_values_videos"].numel()
            * stored["pixel_values_videos"].element_size()
            == inputs["pixel_values_videos"].numel()
            * inputs["pixel_values_videos"].element_size() // 2)


def test_pirl_recompute_evaluates_only_the_selected_transition():
    server = subject.GRPOTrainerServer.__new__(subject.GRPOTrainerServer)
    server.a = SimpleNamespace(sampler="pirl", eta=0.1, num_steps=5,
                               grad_checkpoint=False)
    server.model = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32)
    server.wrappers = []
    state = torch.zeros(1, 1, 1)
    action_mask = torch.ones(1, 2, 3)
    server._split = lambda inputs: (state, action_mask, {})
    weight = torch.nn.Parameter(torch.tensor(0.25))
    calls = []

    def velocity_field(x, t):
        calls.append(float(t[0, 0, 0]))
        return x * weight

    def build_velocity_field(model, state_value, mask_value, **kwargs):
        return velocity_field, (1, 2, 3), torch.device("cpu"), torch.float32

    xs = [torch.full((1, 2, 3), float(index + 1)) for index in range(6)]
    selected = 2
    chunk = {
        "inputs_cpu": {}, "xs_cpu": xs,
        "exec_mask_cpu": torch.ones(2, 3, dtype=torch.bool),
        "skill": "grasp", "eta": 0.1, "denoise_index": selected,
        "pirl_stds": [0.0, 0.0, 0.05, 0.0, 0.0],
    }
    with mock.patch.object(subject, "build_velocity_field", build_velocity_field), \
         mock.patch.object(subject, "set_active_skill"):
        terms = server._forward_new_logp_terms(chunk)
        terms.mean().backward()

    assert len(calls) == 1
    expected_t = 1.0 - float(subject.openpi_timesteps(5)[selected])
    assert abs(calls[0] - expected_t) < 1e-6
    assert terms.shape == (6,)
    assert weight.grad is not None and torch.isfinite(weight.grad)


def test_compatible_chunks_share_one_batched_velocity_forward():
    server = subject.GRPOTrainerServer.__new__(subject.GRPOTrainerServer)
    server.a = SimpleNamespace(sampler="pirl", eta=0.1, num_steps=5,
                               grad_checkpoint=False, microbatch_size=2,
                               batch_model_forward=True)
    server.model = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32)
    server.wrappers = []
    weight = torch.nn.Parameter(torch.tensor(0.25))
    calls = []

    def velocity_field(x, t):
        calls.append(tuple(x.shape))
        return x * weight

    def build_velocity_field(model, state, mask, **kwargs):
        return velocity_field, tuple(mask.shape), torch.device("cpu"), torch.float32

    def make_chunk(offset):
        return {
            "inputs_cpu": {
                "state": torch.full((1, 1, 1), float(offset)),
                "action_mask": torch.ones(1, 2, 3),
                "input_ids": torch.ones(1, 4, dtype=torch.long),
            },
            "xs_cpu": [torch.full((1, 2, 3), float(index + offset + 1))
                       for index in range(6)],
            "exec_mask_cpu": torch.ones(2, 3, dtype=torch.bool),
            "skill": "grasp", "eta": 0.1, "denoise_index": 2,
            "pirl_stds": [0.0, 0.0, 0.05, 0.0, 0.0],
        }

    chunks = [make_chunk(0), make_chunk(10)]
    with mock.patch.object(subject, "build_velocity_field", build_velocity_field), \
         mock.patch.object(subject, "set_active_skill"):
        terms = server._forward_new_logp_terms_batch(chunks)
        sum(item.mean() for item in terms).backward()

    assert calls == [(2, 2, 3)]
    assert [tuple(item.shape) for item in terms] == [(6,), (6,)]
    assert weight.grad is not None and torch.isfinite(weight.grad)
    batch_terms = [item.detach().clone() for item in terms]
    batch_grad = weight.grad.detach().clone()
    weight.grad = None
    calls.clear()
    with mock.patch.object(subject, "build_velocity_field", build_velocity_field), \
         mock.patch.object(subject, "set_active_skill"):
        serial_terms = [server._forward_new_logp_terms(chunk) for chunk in chunks]
        sum(item.mean() for item in serial_terms).backward()
    assert calls == [(1, 2, 3), (1, 2, 3)]
    assert all(torch.allclose(left, right, atol=1e-6)
               for left, right in zip(batch_terms, serial_terms))
    assert torch.allclose(batch_grad, weight.grad, atol=1e-6)


if __name__ == "__main__":
    test_rollout_inputs_store_only_model_visible_precision()
    test_pirl_recompute_evaluates_only_the_selected_transition()
    test_compatible_chunks_share_one_batched_velocity_forward()
    print("selected pi-RL recompute test passed")
