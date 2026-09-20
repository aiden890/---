#!/usr/bin/env python3
"""High-risk MiBoT/RLinf policy contract test using the real Flow-SDE equations.

This intentionally avoids loading the checkpoint.  A tiny differentiable velocity field
exercises the rollout -> trajectory stack/flatten -> learner recompute boundary against
the pinned RLinf image and the production ``pirl_flow_sde`` implementation.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mibot_rlinf_policy import MiBoTRLinfPolicy  # noqa: E402
from rlinf.data.schema.embodied_trajectory_builder import (  # noqa: E402
    stack_list_of_dict_tensor,
)
from rlinf.models.embodiment.base_policy import BasePolicy  # noqa: E402
from rlinf.utils.utils import collect_param_names_need_sync  # noqa: E402
from lora import inject_per_skill_lora, select_trainable  # noqa: E402


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.125))


class _TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        block = nn.Module()
        block.qkv_proj = nn.Linear(4, 12)
        self.dit = nn.Module()
        self.dit.layers = nn.ModuleList([block])


class _FakeProcessor:
    def apply_chat_template(self, conversations, *, state, **kwargs):
        del kwargs
        batch = len(conversations)
        return {
            "state": torch.as_tensor(state, dtype=torch.float32),
            "action_mask": torch.ones(batch, 16, 60, dtype=torch.bool),
            "input_ids": torch.arange(3).repeat(batch, 1),
        }

    def decode_action(self, actions, *, robot_type):
        assert robot_type == "robocasa365"
        return actions


class _CheckpointLikeProcessor(_FakeProcessor):
    """Mirror the real checkpoint bug: action_mask defaults to batch size one."""

    def apply_chat_template(self, conversations, *, state, **kwargs):
        result = super().apply_chat_template(
            conversations, state=state, **kwargs
        )
        result["action_mask"] = result["action_mask"][:1]
        batch = len(conversations)
        result["pixel_values_videos"] = torch.zeros(batch * 5, 8)
        result["video_grid_thw"] = torch.ones(batch * 3, 3, dtype=torch.long)
        return result


def _build_policy() -> MiBoTRLinfPolicy:
    policy = MiBoTRLinfPolicy.__new__(MiBoTRLinfPolicy)
    nn.Module.__init__(policy)
    policy.cfg = {}
    policy.policy = _TinyModel()
    policy.processor = _FakeProcessor()
    policy.lora_wrappers = []
    policy.active_skill = "grasp"
    policy.robot_type = "robocasa365"
    policy.num_steps = 5
    policy.num_action_chunks = 16
    policy.action_dim = 12
    policy.noise_level = 0.1
    policy.denoise_index = 2
    policy.obs_history = 4
    policy.crop = 1.0
    return policy


class RLinfPolicyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        flow_policy = types.ModuleType("flow_policy")

        def build_velocity_field(model, state, action_mask, **kwargs):
            del action_mask, kwargs
            shape = (state.shape[0], 16, 60)

            def velocity(x, time):
                del time
                return torch.ones_like(x) * model.scale

            return velocity, shape, state.device, torch.float32

        flow_policy.build_velocity_field = build_velocity_field
        cls.old_flow = sys.modules.get("flow_policy")
        sys.modules["flow_policy"] = flow_policy

        lora = types.ModuleType("lora")
        lora.set_active_skill = lambda wrappers, skill: None
        cls.old_lora = sys.modules.get("lora")
        sys.modules["lora"] = lora

    @classmethod
    def tearDownClass(cls):
        for name, old in (("flow_policy", cls.old_flow), ("lora", cls.old_lora)):
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old

    def test_rollout_stack_flatten_and_recompute_are_consistent(self):
        torch.manual_seed(7)
        policy = _build_policy()
        batch = 2
        env_obs = {
            "states": torch.zeros(batch, 4, 60),
            "main_images": torch.zeros(batch, 4, 2, 2, 3, dtype=torch.uint8),
            "extra_view_images": torch.zeros(
                batch, 4, 1, 2, 2, 3, dtype=torch.uint8
            ),
            "wrist_images": torch.zeros(batch, 4, 2, 2, 3, dtype=torch.uint8),
            "task_descriptions": ["close lid"] * batch,
        }

        actions, rollout = policy.predict_action_batch(env_obs, mode="train")
        self.assertEqual(tuple(actions.shape), (batch, 16, 12))
        self.assertEqual(tuple(rollout["prev_logprobs"].shape), (batch, 16, 12))
        self.assertEqual(
            tuple(rollout["forward_inputs"]["chains"].shape),
            (batch, 6, 16, 60),
        )

        recomputed = policy.default_forward(rollout["forward_inputs"])["logprobs"]
        torch.testing.assert_close(recomputed, rollout["prev_logprobs"], rtol=0, atol=1e-5)

        # This is the exact operation performed by RLinf's trajectory builder and actor:
        # stack rollout time before batch, then flatten those two dimensions for training.
        stacked = stack_list_of_dict_tensor(
            [rollout["forward_inputs"], rollout["forward_inputs"]]
        )
        flat = {
            key: value.reshape(-1, *value.shape[2:])
            for key, value in stacked.items()
        }
        flat_output = policy.default_forward(flat)["logprobs"]
        self.assertEqual(tuple(flat_output.shape), (2 * batch, 16, 12))

        # A learner-side parameter change must alter the likelihood and retain a usable
        # gradient; otherwise patch sync can appear healthy while GRPO cannot learn.
        with torch.no_grad():
            policy.policy.scale.add_(0.05)
        changed = policy.default_forward(rollout["forward_inputs"])["logprobs"]
        self.assertFalse(torch.equal(changed, rollout["prev_logprobs"]))
        (-changed.mean()).backward()
        self.assertIsNotNone(policy.policy.scale.grad)
        self.assertTrue(torch.isfinite(policy.policy.scale.grad))
        self.assertGreater(abs(float(policy.policy.scale.grad)), 0.0)

    def test_generic_rlinf_eval_signal_disables_sde_logprobs(self):
        torch.manual_seed(11)
        policy = _build_policy()
        env_obs = {
            "states": torch.zeros(1, 4, 60),
            "main_images": torch.zeros(1, 4, 2, 2, 3, dtype=torch.uint8),
            "extra_view_images": torch.zeros(1, 4, 1, 2, 2, 3, dtype=torch.uint8),
            "wrist_images": torch.zeros(1, 4, 2, 2, 3, dtype=torch.uint8),
            "task_descriptions": ["close lid"],
        }
        _, result = policy.predict_action_batch(env_obs, do_sample=False)
        self.assertTrue(torch.equal(result["prev_logprobs"], torch.zeros(1, 16, 12)))

    def test_checkpoint_action_mask_is_expanded_to_rlinf_batch(self):
        policy = _build_policy()
        policy.processor = _CheckpointLikeProcessor()
        batch = 2
        env_obs = {
            "states": torch.zeros(batch, 4, 60),
            "main_images": torch.zeros(batch, 4, 2, 2, 3, dtype=torch.uint8),
            "extra_view_images": torch.zeros(
                batch, 4, 1, 2, 2, 3, dtype=torch.uint8
            ),
            "wrist_images": torch.zeros(batch, 4, 2, 2, 3, dtype=torch.uint8),
            "task_descriptions": ["close lid"] * batch,
        }
        inputs = policy._processor_inputs(env_obs)
        self.assertEqual(tuple(inputs["action_mask"].shape), (batch, 16, 60))
        self.assertFalse(inputs["action_mask"]._is_view())
        packed = policy._pack_model_inputs(inputs, batch)
        self.assertEqual(tuple(packed["pixel_values_videos"].shape), (batch, 5, 8))
        self.assertEqual(tuple(packed["video_grid_thw"].shape), (batch, 3, 3))
        unpacked = policy._unpack_model_inputs(packed)
        self.assertEqual(tuple(unpacked["pixel_values_videos"].shape), (batch * 5, 8))
        self.assertEqual(tuple(unpacked["video_grid_thw"].shape), (batch * 3, 3))

    def test_policy_and_custom_lora_match_rlinf_sync_contract(self):
        self.assertTrue(issubclass(MiBoTRLinfPolicy, BasePolicy))

        actor_model = _TinyDiT()
        rollout_model = _TinyDiT()
        actor_wrappers = inject_per_skill_lora(
            actor_model, skills=("grasp",), rank=2, alpha=4, targets=("qkv_proj",)
        )
        rollout_wrappers = inject_per_skill_lora(
            rollout_model, skills=("grasp",), rank=2, alpha=4, targets=("qkv_proj",)
        )
        select_trainable(actor_model, actor_wrappers, "adapter_only")
        select_trainable(rollout_model, rollout_wrappers, "adapter_only")

        # PatchWeightSyncer rejects actor/rollout key-set mismatches during init.
        self.assertEqual(set(actor_model.state_dict()), set(rollout_model.state_dict()))
        names = collect_param_names_need_sync(actor_model)
        self.assertEqual(
            set(names),
            {
                "dit.layers.0.qkv_proj.lora_A.grasp",
                "dit.layers.0.qkv_proj.lora_B.grasp",
            },
        )


if __name__ == "__main__":
    unittest.main()
