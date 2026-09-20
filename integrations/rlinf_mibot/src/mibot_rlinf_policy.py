"""RLinf-native MiBoT policy for rollout, GRPO recompute, LoRA, and weight sync.

RLinf owns worker placement, FSDP/optimizer execution, advantage computation, and patch
weight synchronization. This class only owns the Xiaomi-specific policy contract:

* RoboCasa365 observation -> pinned processor input
* MiBoT velocity field -> faithful pi-RL Flow-SDE sample
* stored latent path -> differentiable elementwise transition log-probability
* one active per-skill LoRA adapter

The first native milestone intentionally permits exactly one adapter skill per run.
``PerSkillLoRALinear`` stores its active skill as module state, so silently mixing skills
inside one FSDP micro-batch would apply the wrong adapter. Separate GRASP/MOVE/PLACE runs
remain supported and preserve the existing adapter layout.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from rlinf.models.embodiment.base_policy import BasePolicy

from mibot_adapter import ACTIVE_ACTION_DIM, FULL_ACTION_DIM, STATE_DIM


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_numpy_image(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class MiBoTRLinfPolicy(nn.Module, BasePolicy):
    """MiBoT implementation of RLinf's embodied ``BasePolicy`` contract."""

    def __init__(self, cfg: Any, torch_dtype: torch.dtype | None = None):
        super().__init__()
        from transformers import AutoModel, AutoProcessor
        from lora import SKILLS, inject_per_skill_lora, select_trainable, set_active_skill

        self.cfg = cfg
        self.model_path = str(_cfg_get(cfg, "model_path", "/checkpoint"))
        self.robot_type = str(_cfg_get(cfg, "robot_type", "robocasa365"))
        self.num_steps = int(_cfg_get(cfg, "num_steps", 5))
        self.num_action_chunks = int(_cfg_get(cfg, "num_action_chunks", 16))
        self.action_dim = int(_cfg_get(cfg, "action_dim", ACTIVE_ACTION_DIM))
        self.noise_level = float(_cfg_get(cfg, "noise_level", 0.1))
        self.denoise_index = int(_cfg_get(cfg, "denoise_index", self.num_steps // 2))
        self.obs_history = int(_cfg_get(cfg, "obs_history", 4))
        self.crop = float(_cfg_get(cfg, "crop", 0.9))
        if not 0 <= self.denoise_index < self.num_steps:
            raise ValueError(
                f"denoise_index must be in [0, {self.num_steps}), got {self.denoise_index}"
            )
        if self.action_dim != ACTIVE_ACTION_DIM:
            raise ValueError(
                f"MiBoT RoboCasa365 requires action_dim={ACTIVE_ACTION_DIM}, got {self.action_dim}"
            )

        skills_raw = _cfg_get(cfg, "adapter_skills", ["grasp"])
        if isinstance(skills_raw, str):
            skills = tuple(s.strip() for s in skills_raw.split(",") if s.strip())
        else:
            skills = tuple(str(s) for s in skills_raw)
        if len(skills) != 1:
            raise ValueError(
                "RLinf-native MiBoT currently requires exactly one adapter skill per run; "
                f"got {skills}. Launch separate GRASP/MOVE/PLACE actor runs."
            )
        if skills[0] not in SKILLS:
            raise ValueError(f"unknown MiBoT adapter skill {skills[0]!r}; expected one of {SKILLS}")
        self.active_skill = skills[0]

        dtype = torch_dtype or torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.policy = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            attn_implementation=str(_cfg_get(cfg, "attn_implementation", "flash_attention_2")),
            torch_dtype=dtype,
        )

        targets_raw = _cfg_get(cfg, "lora_targets", ["qkv_proj"])
        targets = (
            tuple(s.strip() for s in targets_raw.split(",") if s.strip())
            if isinstance(targets_raw, str)
            else tuple(str(s) for s in targets_raw)
        )
        self.lora_wrappers = inject_per_skill_lora(
            self.policy,
            skills=skills,
            rank=int(_cfg_get(cfg, "lora_rank", 8)),
            alpha=int(_cfg_get(cfg, "lora_alpha", 32)),
            targets=targets,
        )
        if not self.lora_wrappers:
            raise RuntimeError(f"LoRA target selection matched no MiBoT Linear modules: {targets}")
        self.trainable_count, self.trainable_groups = select_trainable(
            self.policy,
            self.lora_wrappers,
            str(_cfg_get(cfg, "train_mode", "adapter_only")),
        )
        set_active_skill(self.lora_wrappers, self.active_skill)
        self._force_dropout_eval()

    @property
    def device(self) -> torch.device:
        return next(self.policy.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.policy.parameters()).dtype

    def _force_dropout_eval(self) -> None:
        # Importance ratios are invalid if rollout and recompute see different dropout.
        for module in self.policy.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._force_dropout_eval()
        return self

    def forward(self, forward_type=None, **kwargs):
        # RLinf calls the default path without an explicit ForwardType for GRPO.
        if forward_type is None or getattr(forward_type, "value", "default") == "default":
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"MiBoT does not support forward_type={forward_type!r}")

    def _frame_history(self, value: Any, batch_index: int) -> list[np.ndarray]:
        arr = _as_numpy_image(value)
        sample = arr[batch_index]
        # A caller may provide explicit [B,T,H,W,C] history. RLinf's stock RoboCasa365
        # env currently provides [B,H,W,C], in which case repeat the current frame to
        # preserve the checkpoint's video-shaped input contract.
        if sample.ndim == 4:
            frames = [sample[i] for i in range(sample.shape[0])]
        else:
            frames = [sample]
        frames = frames[-self.obs_history :]
        while len(frames) < self.obs_history:
            frames.insert(0, frames[0])
        try:
            import rollout

            return [rollout.center_crop(frame, self.crop) for frame in frames]
        except ImportError:
            return frames

    def _right_camera(self, env_obs: dict[str, Any]) -> Any:
        extra = env_obs.get("extra_view_images")
        if extra is None:
            return env_obs["main_images"]
        if torch.is_tensor(extra):
            # Stock RoboCasa365: [B, num_extra_views, H, W, C].
            # MiBoT history wrapper: [B, T, num_extra_views, H, W, C].
            return extra[:, :, 0] if extra.ndim == 6 else extra[:, 0]
        extra_np = np.asarray(extra)
        return extra_np[:, :, 0] if extra_np.ndim == 6 else extra_np[:, 0]

    def _processor_inputs(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        states = env_obs["states"]
        if torch.is_tensor(states):
            states_np = states.detach().cpu().float().numpy()
        else:
            states_np = np.asarray(states, dtype=np.float32)
        batch_size = int(states_np.shape[0])
        right_images = self._right_camera(env_obs)
        wrist_images = env_obs.get("wrist_images")
        if wrist_images is None:
            wrist_images = env_obs["main_images"]
        prompts = env_obs.get("task_descriptions") or [""] * batch_size

        conversations = []
        state_batch = np.zeros(
            (batch_size, self.obs_history, STATE_DIM), dtype=np.float32
        )
        for i in range(batch_size):
            state_i = states_np[i]
            if state_i.ndim == 1:
                state_history = np.repeat(state_i[None], self.obs_history, axis=0)
            else:
                state_history = state_i.reshape(state_i.shape[0], -1)[-self.obs_history :]
                if state_history.shape[0] < self.obs_history:
                    state_history = np.concatenate(
                        [
                            np.repeat(
                                state_history[:1],
                                self.obs_history - state_history.shape[0],
                                axis=0,
                            ),
                            state_history,
                        ],
                        axis=0,
                    )
            width = min(state_history.shape[-1], STATE_DIM)
            state_batch[i, :, :width] = state_history[:, :width]
            left = self._frame_history(env_obs["main_images"], i)
            right = self._frame_history(right_images, i)
            wrist = self._frame_history(wrist_images, i)
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Left camera: "},
                            {"type": "video", "video": left},
                            {"type": "text", "text": "\nRight camera: "},
                            {"type": "video", "video": right},
                            {"type": "text", "text": "\nWrist camera: "},
                            {"type": "video", "video": wrist},
                            {
                                "type": "text",
                                "text": (
                                    "\n\nGenerate robot actions for the task:\n"
                                    f"{prompts[i]} /no_cot"
                                ),
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "<cot></cot>"}],
                    },
                ]
            )

        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            do_resize=False,
            state=state_batch,
            robot_type=self.robot_type,
        )
        tensor_inputs = {
            key: value for key, value in dict(inputs).items() if torch.is_tensor(value)
        }
        # The checkpoint's MiBotProcessor calls get_action_mask(robot_type) without
        # forwarding batch_size, so it always emits [1,L,A] even for batched text/state.
        # RLinf batches environments; expand this robot-constant mask explicitly.
        action_mask = tensor_inputs.get("action_mask")
        if action_mask is None:
            raise RuntimeError("MiBoT processor output is missing action_mask")
        if action_mask.shape[0] == 1 and batch_size > 1:
            tensor_inputs["action_mask"] = action_mask.expand(
                batch_size, *action_mask.shape[1:]
            ).clone()
        elif action_mask.shape[0] != batch_size:
            raise RuntimeError(
                "MiBoT processor action_mask batch does not match observations: "
                f"{action_mask.shape[0]} != {batch_size}"
            )
        return tensor_inputs

    def _to_policy_device(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = {}
        for key, value in inputs.items():
            if value.is_floating_point():
                result[key] = value.to(device=self.device, dtype=self.dtype)
            else:
                result[key] = value.to(device=self.device)
        return result

    @staticmethod
    def _split_inputs(inputs: dict[str, torch.Tensor]):
        data = dict(inputs)
        state = data.pop("state")
        action_mask = data.pop("action_mask")
        return state, action_mask, data

    @staticmethod
    def _pack_model_inputs(
        inputs: dict[str, torch.Tensor], batch_size: int
    ) -> dict[str, torch.Tensor]:
        """Give flattened multimodal processor tensors an explicit RLinf batch axis."""
        packed = dict(inputs)
        for key in ("pixel_values", "pixel_values_videos"):
            value = packed.get(key)
            if value is not None:
                if value.shape[0] % batch_size:
                    raise RuntimeError(
                        f"{key} patch count {value.shape[0]} is not divisible by "
                        f"batch size {batch_size}"
                    )
                packed[key] = value.reshape(batch_size, -1, *value.shape[1:])
        for key in ("image_grid_thw", "video_grid_thw"):
            value = packed.get(key)
            if value is not None:
                if value.shape[0] % batch_size:
                    raise RuntimeError(
                        f"{key} item count {value.shape[0]} is not divisible by "
                        f"batch size {batch_size}"
                    )
                packed[key] = value.reshape(batch_size, -1, *value.shape[1:])
        for key, value in packed.items():
            if value.shape[0] != batch_size:
                raise RuntimeError(
                    f"packed processor tensor {key} has non-batch leading dimension "
                    f"{value.shape[0]} (expected {batch_size})"
                )
        return packed

    @staticmethod
    def _unpack_model_inputs(
        inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Restore checkpoint processor shapes after RLinf split/stack/flatten."""
        unpacked = dict(inputs)
        for key in ("pixel_values", "pixel_values_videos"):
            value = unpacked.get(key)
            if value is not None and value.ndim >= 3:
                unpacked[key] = value.reshape(-1, *value.shape[2:])
        for key in ("image_grid_thw", "video_grid_thw"):
            value = unpacked.get(key)
            if value is not None and value.ndim >= 3:
                unpacked[key] = value.reshape(-1, *value.shape[2:])
        return unpacked

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        compute_values: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del compute_values
        import pirl_flow_sde
        from flow_policy import build_velocity_field
        from lora import set_active_skill

        set_active_skill(self.lora_wrappers, self.active_skill)
        model_inputs = self._to_policy_device(self._processor_inputs(env_obs))
        state, action_mask, vlm_inputs = self._split_inputs(model_inputs)
        velocity, shape, device, dtype = build_velocity_field(
            self.policy, state, action_mask, **vlm_inputs
        )
        # RLinf's pinned HuggingFace worker passes ``mode`` only for a hard-coded list
        # of built-in model enums. Dynamically registered models still receive its
        # train/eval sampling dictionary, where eval is represented by do_sample=False.
        # Honour both paths so evaluation cannot silently use training SDE noise.
        if kwargs.get("do_sample") is False:
            mode = "eval"
        noise_level = self.noise_level if mode == "train" else 0.0
        selected = self.denoise_index if noise_level > 0.0 else None
        sampled = pirl_flow_sde.pirl_flow_sde_sample(
            velocity,
            shape,
            num_steps=self.num_steps,
            noise_level=noise_level,
            device=device,
            dtype=dtype,
            denoise_index=selected,
        )
        decoded = self.processor.decode_action(sampled.actions, robot_type=self.robot_type)
        if not torch.is_tensor(decoded):
            decoded = torch.as_tensor(decoded, device=device)
        actions = decoded[..., : self.action_dim].float().contiguous()
        batch_size = actions.shape[0]

        if selected is None:
            old_logprobs = torch.zeros(
                (batch_size, self.num_action_chunks, self.action_dim),
                device=device,
                dtype=torch.float32,
            )
        else:
            old_logprobs = sampled.perstep_perdim_logprob[
                :, selected, : self.num_action_chunks, : self.action_dim
            ].float().contiguous()

        forward_inputs = {
            "chains": torch.stack(sampled.xs, dim=1).contiguous(),
            "denoise_inds": torch.full(
                (batch_size, 1), self.denoise_index, device=device, dtype=torch.long
            ),
            "action": actions[:, : self.num_action_chunks].reshape(batch_size, -1),
            "model_action": sampled.actions.reshape(batch_size, -1).contiguous(),
        }
        # RLinf's trajectory builder stacks forward_inputs one dictionary level deep;
        # nested dictionaries are not part of that schema. Prefix processor tensors so
        # they survive stack/split/replay unchanged.
        packed_inputs = self._pack_model_inputs(model_inputs, batch_size)
        forward_inputs.update({
            f"model_input__{key}": value.contiguous()
            for key, value in packed_inputs.items()
        })
        return actions, {
            "prev_logprobs": old_logprobs,
            "prev_values": torch.zeros((batch_size, 1), device=device),
            "forward_inputs": forward_inputs,
            "model_actions": sampled.actions,
        }

    def default_forward(
        self, forward_inputs: dict[str, Any], **kwargs
    ) -> dict[str, torch.Tensor]:
        del kwargs
        import pirl_flow_sde
        from flow_policy import build_velocity_field
        from lora import set_active_skill

        set_active_skill(self.lora_wrappers, self.active_skill)
        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]
        if not torch.all(denoise_inds == self.denoise_index):
            raise RuntimeError(
                "MiBoT native actor received mixed denoise indices; use the fixed configured index"
            )
        model_inputs = {
            key[len("model_input__") :]: value
            for key, value in forward_inputs.items()
            if key.startswith("model_input__")
        }
        if not model_inputs:
            raise RuntimeError("MiBoT forward_inputs are missing prefixed processor tensors")
        model_inputs = self._unpack_model_inputs(model_inputs)
        state, action_mask, vlm_inputs = self._split_inputs(model_inputs)
        velocity, shape, device, dtype = build_velocity_field(
            self.policy, state, action_mask, **vlm_inputs
        )

        idx = self.denoise_index
        timesteps = pirl_flow_sde.openpi_timesteps(self.num_steps, device=device)
        sigmas = pirl_flow_sde.openpi_sigmas(
            self.num_steps, self.noise_level, device=device
        )
        t_o = float(timesteps[idx].item())
        t_next = float(timesteps[idx + 1].item())
        t_m = torch.full(
            (shape[0], 1, 1), 1.0 - t_o, device=device, dtype=dtype
        )
        x_t = chains[:, idx]
        velocity_value = velocity(x_t, t_m)
        selected_mean, _ = pirl_flow_sde.pirl_step_mean_std(
            x_t, velocity_value, t_o, t_next, float(sigmas[idx].item())
        )
        means = [torch.zeros_like(x_t) for _ in range(self.num_steps)]
        means[idx] = selected_mean
        xs = list(chains.unbind(dim=1))
        stds = [0.0] * self.num_steps
        stds[idx] = float((torch.sqrt(timesteps[idx] - timesteps[idx + 1]) * sigmas[idx]).item())
        logprobs = pirl_flow_sde.pirl_transition_logprob_elements(
            xs, means, stds, idx
        )[:, : self.num_action_chunks, : self.action_dim]
        return {
            "logprobs": logprobs.float().contiguous(),
            "values": torch.zeros(shape[0], device=device, dtype=torch.float32),
            "entropy": torch.zeros((shape[0], 1), device=device, dtype=torch.float32),
        }


def build_mibot_policy(cfg: Any, torch_dtype: torch.dtype | None = None) -> MiBoTRLinfPolicy:
    """RLinf registry builder signature: ``(DictConfig, dtype) -> model``."""
    return MiBoTRLinfPolicy(cfg, torch_dtype=torch_dtype)


__all__ = ["MiBoTRLinfPolicy", "build_mibot_policy"]
