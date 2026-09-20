"""MiBoT <-> RLinf integration skeleton.

This module is the clean interface layer between the pinned Xiaomi MiBoT checkpoint and
the RLinf training framework. It deliberately REUSES the already-verified in-repo
components by import (never re-implements them):

  * flow_policy.build_velocity_field  (rl-env-t_4f3f2b20/src/flow_policy.py)
      -> reproduces the checkpoint's conditioning prep and exposes dit_forward as a
         velocity field. THE one place that mirrors upstream internals; asserted bit-exact.
  * pirl_flow_sde.pirl_flow_sde_sample / pirl_transition_logprob
      (rl-env-t_4f3f2b20/src/pirl_flow_sde.py)
      -> the VERIFIED faithful pi-RL Flow-SDE (RLinf@bde6c918 equations). We do NOT
         re-derive the corrected drift / sigma schedule / log-prob here.
  * rollout.observation_to_state / collect_images / EvalClient message build
      (xiaomi-cu121/rollout.py)  -> observation transforms & the 14D EE-first state.
  * reward.py + skill_manager.py (rl-env card) -> reward definition + skill FSM.

On the deployment server these paths are mounted at /rl_env/src and /skill_eval_tools;
locally they resolve via the repo tree. Everything here is import-time safe WITHOUT a GPU
(the heavy objects are created lazily inside methods), so static import tests pass on
lab-desktop while the actual model work happens only inside the GPU container.

The RLinf-native policy and registry bridge live in ``mibot_rlinf_policy.py`` and
``rlinf_env.py``.  GPU execution still requires the checkpoint and simulator image, but
the framework registration is no longer a placeholder.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# --- resolve the reused verified components (mounted or in-repo) ----------------------
def _find_repo_path(relative: Path) -> Path | None:
    """Find a sibling component without assuming the integration's install depth."""
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / relative
        if candidate.exists():
            return candidate
    return None


_CANDIDATE_RL_ENV = [Path("/rl_env/src"), _find_repo_path(Path("rl-env-t_4f3f2b20/src"))]
_CANDIDATE_SKILL = [
    Path("/skill_eval_tools"),
    _find_repo_path(Path("rollouts-xiaomi-t_4a072806/tools")),
]
_CANDIDATE_ROLLOUT = [Path("/work"), _find_repo_path(Path("xiaomi-cu121"))]
_CANDIDATE_TRAIN = [Path("/train/src"), _find_repo_path(Path("rl-train-t_3ed65912/src"))]
for cands in (
    _CANDIDATE_RL_ENV,
    _CANDIDATE_SKILL,
    _CANDIDATE_ROLLOUT,
    _CANDIDATE_TRAIN,
):
    for p in cands:
        if p is not None and p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

# 12 active action dims (RoboCasa365 EE-first: 0-11 mean0/std1 -> raw action[12] is the
# CFM target directly, no normalization). The checkpoint action tensor is 60-D but only
# the first 12 dims are active for CloseBlenderLid.
ACTIVE_ACTION_DIM = 12
FULL_ACTION_DIM = 60
STATE_DIM = 60          # padded; first 14 are the EE-first RoboCasa365 state
DEFAULT_NUM_STEPS = 5   # MiBoT deterministic Euler steps (matches checkpoint sampler)


@dataclass
class MiBoTConfig:
    checkpoint_dir: str = "/checkpoint"
    device: str = "cuda"
    dtype: str = "bfloat16"
    num_steps: int = DEFAULT_NUM_STEPS
    replan_steps: int = 16          # executed chunk length before replanning
    active_action_dim: int = ACTIVE_ACTION_DIM
    # adapter-only training: freeze VLM + action-expert body, train adapters + term head.
    train_mode: str = "adapter_only"


class MiBoTModel:
    """Lazy wrapper around the pinned MiBoT checkpoint.

    All heavy work (weight load, CUDA) is deferred to .load() so importing this module
    needs no GPU. The forward/velocity path reuses flow_policy.build_velocity_field.
    """

    def __init__(self, cfg: MiBoTConfig):
        self.cfg = cfg
        self.model: Any = None
        self.processor: Any = None
        self._loaded = False

    # --- interface 1: model + processor loading (GPU, NEEDS_SERVER) --------------------
    def load(self):
        """Load checkpoint + processor. Verified interface; requires GPU + /checkpoint."""
        import torch
        from transformers import AutoModel, AutoProcessor
        dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[self.cfg.dtype]
        self.processor = AutoProcessor.from_pretrained(self.cfg.checkpoint_dir, trust_remote_code=True)
        self.model = (AutoModel.from_pretrained(self.cfg.checkpoint_dir, trust_remote_code=True, torch_dtype=dt)
                      .to(self.cfg.device).eval())
        # adapter-only: freeze all base params; adapters/term-head are attached by the
        # trainer (see rl-train card lora.py) — kept here as an explicit contract.
        if self.cfg.train_mode == "adapter_only":
            for p in self.model.parameters():
                p.requires_grad_(False)
        self._loaded = True
        return self

    # --- interface 2: observation transforms (reused from rollout.py) -----------------
    @staticmethod
    def observation_transforms():
        """Return (observation_to_state, collect_images) reused from the verified rollout.

        These build the 14D EE-first RoboCasa365 state and the 3-camera image dict exactly
        as the checkpoint was fed during the verified zero-shot rollouts.
        """
        import rollout  # xiaomi-cu121/rollout.py
        return rollout.observation_to_state, rollout.collect_images

    def build_messages(self, image_history: dict, instruction: str):
        """Reuse the checkpoint processor's message format (single source of truth)."""
        # EvalClient._build_messages is the canonical prompt; construct a throwaway client
        # only for its formatting helper without opening a network connection.
        import rollout
        return rollout.EvalClient._build_messages(self, image_history, instruction)  # type: ignore

    # --- interface 3: velocity field (reuses flow_policy) -----------------------------
    def velocity_field(self, state, action_mask, **model_kwargs) -> tuple[Callable, tuple, Any, Any]:
        """Return (velocity_field, shape, device, dtype) from the checkpoint's own forward.

        Delegates to the verified flow_policy.build_velocity_field — the ONLY place that
        mirrors MiBoTForActionGeneration.forward internals.
        """
        if not self._loaded:
            raise RuntimeError("call .load() first (GPU required)")
        import flow_policy
        return flow_policy.build_velocity_field(self.model, state, action_mask, **model_kwargs)

    # --- interface 4: 12-D action decode ----------------------------------------------
    @staticmethod
    def decode_action(action_chunk):
        """Slice the checkpoint's 60-D action tensor down to the 12 active RoboCasa dims.

        action_chunk: [..., FULL_ACTION_DIM] -> [..., ACTIVE_ACTION_DIM]. No de-norm needed
        (active dims are mean0/std1; raw action[:12] is the env-space command directly).
        """
        return action_chunk[..., :ACTIVE_ACTION_DIM]


class MiBoTSampler:
    """Bridges the MiBoT velocity field to the VERIFIED pi-RL Flow-SDE sampler.

    Provides both the RL rollout path (noise_level>0, stores per-step stds + latent path)
    and the deterministic ODE eval path (noise_level=0, bit-exact with the checkpoint).
    Zero equation code lives here — it calls pirl_flow_sde.* directly.
    """

    def __init__(self, model: MiBoTModel):
        self.model = model

    def sample(self, state, action_mask, *, noise_level: float = 0.0,
               num_steps: Optional[int] = None, generator=None, x0=None,
               noise_sequence=None, **model_kwargs):
        import pirl_flow_sde
        vfield, shape, device, dtype = self.model.velocity_field(state, action_mask, **model_kwargs)
        return pirl_flow_sde.pirl_flow_sde_sample(
            vfield, shape, num_steps=num_steps or self.model.cfg.num_steps,
            noise_level=noise_level, device=device, dtype=dtype,
            generator=generator, x0=x0, noise_sequence=noise_sequence,
        )

    @staticmethod
    def recompute_logprob(xs, means, stds, executed_mask=None):
        """PPO/GRPO recompute path — reuse the verified pirl_transition_logprob."""
        import pirl_flow_sde
        return pirl_flow_sde.pirl_transition_logprob(xs, means, stds, executed_mask=executed_mask)

    def ode_action(self, state, action_mask, **model_kwargs):
        """Deterministic ODE eval: noise_level=0 -> equals the checkpoint Euler ODE.

        Returns the decoded 12-D action chunk. This is the explicit deterministic eval path
        the acceptance gate compares before/after training on identical seeds.
        """
        res = self.sample(state, action_mask, noise_level=0.0, **model_kwargs)
        return MiBoTModel.decode_action(res.actions)


# --- interface 5: reward / env adapter (reuse reward.py + skill_manager.py) ------------
def load_reward_manager(use_milestones: bool = False):
    """Return a RewardManager built from the VERIFIED reward definition.

    Default use_milestones=False = pay-once-at-the-end (the user's chosen payout mode).
    """
    import reward
    cfg = reward.RewardConfig(use_milestones=use_milestones)
    return reward.RewardManager(cfg)


def load_skill_monitor(skill: str):
    """Return the per-skill SUCCESS/DROPPED/TIMEOUT FSM from the verified skill_manager."""
    import skill_manager
    return skill_manager.SkillMonitor(skill)


# --- interface 6: RLinf registration descriptor ----------------------------------------
@dataclass
class RLinfModelSpec:
    """Descriptor RLinf needs to register MiBoT as an embodied action model.

    The actual ``register_model`` call runs inside the container via
    :func:`rlinf_env.register_mibot`. Kept as data here so the contract is inspectable
    without importing RLinf.
    """
    name: str = "mibot_robocasa365"
    action_dim: int = ACTIVE_ACTION_DIM
    full_action_dim: int = FULL_ACTION_DIM
    state_dim: int = STATE_DIM
    num_cameras: int = 3
    sampler: str = "pirl_flow_sde"
    framework_commit: str = "bde6c918642abf9a4776cb1d5fabcc5087dfe195"
    train_mode: str = "adapter_only"
    extra: dict = field(default_factory=dict)


__all__ = [
    "MiBoTConfig", "MiBoTModel", "MiBoTSampler", "RLinfModelSpec",
    "load_reward_manager", "load_skill_monitor",
    "ACTIVE_ACTION_DIM", "FULL_ACTION_DIM", "STATE_DIM",
]
