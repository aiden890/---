from pathlib import Path
import pickle
import functools
import numpy as np


class Backend:
    def __init__(self, checkpoint, lr=1e-4):
        self.checkpoint = Path(checkpoint)
        import jax
        import jax.numpy as jnp
        import flax.nnx as nnx
        import optax
        from openpi.models import model as base, pi0_config, tokenizer
        from openpi.policies.robocasa_policy import RobocasaInputs, RobocasaOutputs
        from openpi.shared import nnx_utils, normalize
        from openpi import transforms

        self.jax = jax
        self.jnp = jnp
        self.nnx = nnx
        self.optax = optax
        mc = pi0_config.Pi0Config(
            pi05=True, max_token_len=200, action_expert_variant="gemma_300m_lora"
        )
        import flax.traverse_util

        abstract = nnx.eval_shape(mc.create, jax.random.key(42))
        reference = flax.traverse_util.flatten_dict(nnx.state(abstract).to_pure_dict(), sep="/")
        loaded = flax.traverse_util.flatten_dict(
            base.restore_params(Path(checkpoint) / "params", dtype=jnp.bfloat16), sep="/"
        )
        for i, (key, value) in enumerate(reference.items()):
            if "lora" in key:
                # Zero B preserves the pretrained policy exactly before update 1.
                loaded[key] = (
                    jnp.zeros(value.shape, jnp.float32)
                    if key.endswith("lora_b")
                    else jax.random.normal(jax.random.key(42 + i), value.shape) * 0.01
                )
        self.model = mc.load(flax.traverse_util.unflatten_dict(loaded, sep="/"))
        self.filter = nnx.All(nnx.Param, nnx_utils.PathRegex(".*lora.*"))
        params = nnx.state(self.model, self.filter)
        assert len(params.flat_state()) > 0
        self.tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adamw(lr))
        self.opt_state = self.tx.init(params)
        stats = normalize.load(Path(checkpoint) / "assets")
        self.action_mean = np.asarray(stats["actions"].mean)[:12]
        self.constant_dims = np.flatnonzero(np.asarray(stats["actions"].std)[:12] < 1e-5)
        # Exact native transform chain for pi05_pretrain_human300; avoid importing
        # the offline dataset registry and simulator into the gradient process.
        self.transform = transforms.compose(
            [
                RobocasaInputs(mc.action_dim, mc.model_type),
                transforms.Normalize(stats, use_quantiles=False),
                transforms.InjectDefaultPrompt(None),
                transforms.ResizeImages(224, 224),
                transforms.TokenizePrompt(
                    tokenizer.PaligemmaTokenizer(200), discrete_state_input=True
                ),
                transforms.PadStatesAndActions(32),
            ]
        )
        self.output = transforms.compose(
            [transforms.Unnormalize(stats, use_quantiles=False), RobocasaOutputs()]
        )
        self.base = base

        @functools.partial(nnx.jit, donate_argnums=(0, 1))
        def step(model, opt_state, observation, actions, valid, rng):
            def objective(m):
                loss = m.compute_loss(rng, observation, actions, train=True)
                return (loss * valid).sum() / valid.sum()

            loss, grads = nnx.value_and_grad(objective, argnums=nnx.DiffState(0, self.filter))(
                model
            )
            params = nnx.state(model, self.filter)
            updates, opt_state = self.tx.update(grads, opt_state, params)
            nnx.update(model, optax.apply_updates(params, updates))
            return loss, optax.global_norm(grads), opt_state

        self.step = step
        # Native module_jit freezes weights when constructed. Build it only
        # after loading the adapter, and invalidate after every update/reload.
        self.infer_fn = None

    def update(self, sample, seed=0):
        from openpi_client import image_tools

        o = sample["obs"]
        state = np.concatenate(
            [
                o[k]
                for k in [
                    "state.end_effector_position_relative",
                    "state.end_effector_rotation_relative",
                    "state.base_position",
                    "state.base_rotation",
                    "state.gripper_qpos",
                ]
            ]
        )
        actions = sample["actions"].copy()
        for dim in self.constant_dims:
            # The source model's floating-point jitter must not be amplified by
            # a zero training std. Reject real commands instead of silently
            # discarding them; only canonicalize the checkpoint's dead band.
            deviation = np.abs(actions[sample["valid"], dim] - self.action_mean[dim])
            assert deviation.max(initial=0) < 0.02, (
                f"Nontrivial action in zero-std dimension {dim}; recompute compatible normalization stats"
            )
            actions[:, dim] = self.action_mean[dim]
        raw = {"observation/state": state, "prompt": sample["prompt"], "actions": actions}
        for key, obskey in [
            ("observation/image", "video.robot0_agentview_left"),
            ("observation/right_image", "video.robot0_agentview_right"),
            ("observation/wrist_image", "video.robot0_eye_in_hand"),
        ]:
            raw[key] = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(o[obskey], 224, 224)
            )
        data = self.transform(raw)
        data = self.jax.tree.map(lambda x: self.jnp.asarray(x)[None], data)
        obs = self.base.Observation.from_dict(data)
        loss, gn, self.opt_state = self.step(
            self.model,
            self.opt_state,
            obs,
            data["actions"],
            self.jnp.asarray(sample["valid"])[None],
            self.jax.random.key(seed),
        )
        self.infer_fn = None
        loss = float(loss)
        gn = float(gn)
        assert np.isfinite(loss) and np.isfinite(gn)
        return dict(
            loss=loss, grad_norm=gn, canonicalized_zero_std_dims=self.constant_dims.tolist()
        )

    def save(self, path, step):
        state = self.nnx.state(self.model, self.filter).to_pure_dict()
        import flax.traverse_util

        flat = flax.traverse_util.flatten_dict(state, sep="/")
        np.savez(Path(path) / "pi05-lora.npz", **{k: np.asarray(v) for k, v in flat.items()})
        with open(Path(path) / "pi05-optimizer.pkl", "wb") as f:
            pickle.dump(dict(optimizer=self.jax.device_get(self.opt_state), step=step), f)

    def load(self, path):
        import flax.traverse_util

        with np.load(Path(path) / "pi05-lora.npz", allow_pickle=False) as f:
            flat = {k: self.jnp.asarray(f[k]) for k in f.files}
        s = self.nnx.state(self.model, self.filter)
        s.replace_by_pure_dict(flax.traverse_util.unflatten_dict(flat, sep="/"))
        self.nnx.update(self.model, s)
        with open(Path(path) / "pi05-optimizer.pkl", "rb") as f:
            self.opt_state = self.jax.tree.map(self.jnp.asarray, pickle.load(f)["optimizer"])
        self.infer_fn = None

    def parameters(self):
        import flax.traverse_util

        return {
            k: np.asarray(v).copy()
            for k, v in flax.traverse_util.flatten_dict(
                self.nnx.state(self.model, self.filter).to_pure_dict(), sep="/"
            ).items()
        }

    def infer(self, sample, seed=0):
        from openpi_client import image_tools

        o = sample["obs"]
        state = np.concatenate(
            [
                o[k]
                for k in [
                    "state.end_effector_position_relative",
                    "state.end_effector_rotation_relative",
                    "state.base_position",
                    "state.base_rotation",
                    "state.gripper_qpos",
                ]
            ]
        )
        raw = {"observation/state": state, "prompt": sample["prompt"]}
        for key, obskey in [
            ("observation/image", "video.robot0_agentview_left"),
            ("observation/right_image", "video.robot0_agentview_right"),
            ("observation/wrist_image", "video.robot0_eye_in_hand"),
        ]:
            raw[key] = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(o[obskey], 224, 224)
            )
        data = self.transform(raw)
        data = self.jax.tree.map(lambda x: self.jnp.asarray(x)[None], data)
        if self.infer_fn is None:
            from openpi.shared import nnx_utils

            self.infer_fn = nnx_utils.module_jit(self.model.sample_actions)
        pred = self.infer_fn(
            self.jax.random.key(seed), self.base.Observation.from_dict(data), num_steps=10
        )
        return self.output({"actions": np.asarray(pred[0]), "state": np.asarray(data["state"][0])})[
            "actions"
        ]

    def frozen_fingerprint(self):
        import flax.traverse_util
        import hashlib

        state = self.nnx.state(self.model, self.nnx.All(self.nnx.Param, self.nnx.Not(self.filter)))
        flat = flax.traverse_util.flatten_dict(state.to_pure_dict(), sep="/")
        h = hashlib.sha256()
        for name, value in sorted(flat.items()):
            h.update(name.encode())
            a = np.asarray(self.jax.device_get(value))
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        return h.hexdigest()
