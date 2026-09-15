# Goal-3 design — per-skill DiT LoRA flow-matching SFT (plan, pre-GPU)

Task t_e5cd5736, phase 3 design. Written GPU-free while the GR00T eval holds the
v4 GPU; execution runs sequentially after that frees. No new project — reuses the
verified `rl-env-t_4f3f2b20` (flow_sde, flow_policy, skill_eval snapshot/restore)
and `rl-train-t_3ed65912` (lora.py PerSkillLoRALinear, grpo trainer server) by
import only.

## Objective: conditional flow-matching (CFM) SFT, per-skill LoRA

The Xiaomi MiBoT action expert is a rectified-flow model: at deploy it integrates
the deterministic Euler ODE `x_{k+1} = x_k + v_θ(x_k, t_k)·dt` from `x_0 ~ N(0,I)`
(verified bit-for-bit by `rl-env/src/flow_sde.py` at eta=0). SFT is the exact
reverse — regress the velocity field to the demo actions with the CFM loss:

```
x1 = demo action chunk        (target, shape [L, A])
x0 ~ N(0, I)
t  ~ U(0,1)                   (per-example, broadcast over the chunk)
xt = (1 - t)·x0 + t·x1        (linear/rectified-flow interpolant)
u  = x1 - x0                  (target velocity, constant in t for this interpolant)
L  = mean_over_masked( || v_θ(xt, t) - u ||^2 )
```

This matches the CFM objective used by π0/π0.5 and πRL (arXiv 2510.25889) and
ReinFlow (arXiv 2505.22094, NeurIPS 2025); πRL uses LoRA r=32/α=32 for joint VLM+AE
tuning — we cite that for the arm-C hyperparameters. The velocity field `v_θ` is
exactly the checkpoint's `dit_forward` reached through `flow_policy.build_velocity_field`
(the one place mirroring upstream internals, already asserted vs the model's own
forward). We do NOT edit the hash-pinned modeling file.

## Action loss mask (the skill boundary)

Each per-skill training example is a fixed-length action chunk drawn from a source
demo. The CFM squared error is averaged **only over action steps inside the skill's
`[start, end)` span** (from `results/skill_segments.json`); steps outside the span —
and the padding beyond `action_dim` / chunk length — are zeroed in the loss. So the
GRASP adapter sees gradient only from GRASP-labelled action steps, etc. — strict
per-skill parameter isolation, consistent with the existing `PerSkillLoRALinear`
(only the active skill's adapter contributes to the forward).

## Conditioning per arm (Goal 2 / Goal 5)

Instruction text + skill_id come from `src/conditioning.py`:
- `nl_only`   — per-skill NL string, per-skill LoRA key.
- `nl_plus_skill_id` — overall goal + `[id:name]` tag + NL + object/dest/constraints,
  per-skill LoRA key.
- `shared_lora_learned_embedding` — text = overall goal only; ONE shared LoRA + a
  learned `nn.Embedding(3, dit_hidden)` added to the DiT conditioning (state_embed
  stream). Do not presume it wins; select by heldout (Goal 5).

## Verification gate (in order; do not skip ahead) — required before any RL

1. **eta=0 identity re-probe.** Re-run `rl-env/scripts/sde_probe.py`: freshly loaded
   checkpoint, LoRA injected but zero-initialized B ⇒ eta=0 sampler == model.forward
   actions, max_abs_diff = 0.0. Confirms SFT starts exactly at the pretrained policy.
2. **Tiny-overfit.** Train one skill's LoRA on 2–4 demos to near-zero CFM loss;
   assert the loss drops >100x and the sampled chunk matches those demos' action
   spans (per-dim MAE below a small threshold). Proves the loss + mask + optimizer
   path actually learn.
3. **Heldout validation.** CFM loss on the val split (never trained) must be finite
   and below the pretrained-init val loss for that skill; report per-skill.
4. **Valid entry-state rollout gate [GPU/EGL].** Restore each skill's real entry
   state via `skill_eval.Sim.snapshot/restore` (GRASP from reset, MOVE from a restored
   post-grasp state, RELEASE from a restored pre-place state — the verified restore:
   d_restore_qpos=0.0, predicates identical), run the SFT policy for that skill, and
   require the skill's SkillMonitor outcome to be no worse than baseline on a small
   seed set before spending the N=50 budget.

Only after 1–4 pass per skill do Goal 4 (eval) and Goal 6 (RL) run.

## Checkpoint provenance (every SFT checkpoint must carry)

Restoring must reproduce the exact trainable slice (audit lesson: op_save/op_load
must round-trip the FULL trainable set, not just LoRA). Each `.pt` stores:
- `lora_state` — per-skill (or shared) LoRA A/B tensors, keyed by name.
- `extra_trainable` — learned skill embedding (arm C) and any unfrozen slice, by name.
- `optimizer_state`, and the torch/cuda/numpy/python **RNG states**.
- `config` — arm, skill(s), rank/alpha, lr, chunk len, num flow steps, batch, epochs.
- `data_hash` — the `results/data_manifest.json` SHA set (source parquet + meta),
  plus the split seed, so the training data is pinned.
- `git_commit` of this workspace + the pinned model revision
  `3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4`.
A reload-equality test (corrupt all trainables, reload, require eta=0 action ==
pre-save snapshot, d=0) gates the save/load path, reusing the pattern from the
audited trainer.

## Layout / run interface

- `src/sft_dataset.py`  — builds masked per-skill action-chunk examples from the
  manifest + parquet (dataloader; applies the loss mask).
- `src/cfm_sft.py`      — the CFM loss + per-skill LoRA SFT loop; imports
  `rl-env/src/flow_policy.build_velocity_field` and `rl-train/src/lora`.
- `scripts/run-sft.sh`  — subcommands mirroring run-train.sh: `sft-probe` (gate 1),
  `sft-overfit` (gate 2), `sft-train <arm> <skill>` , `sft-val` (gate 3),
  `sft-rollout-gate` (gate 4). Each in its own short-lived container; shared Xiaomi
  server restarted before any rollout; writes to task-scoped `results/` (no overwrite).

## Compute / sequencing note

All of the above needs the GPU (checkpoint forward). It is therefore GATED on the
GR00T comparison eval finishing (do not intrude on its GPU). Gates 1–3 are cheap
(<~1 GPU-hr total for a pilot); gate 4 + Goals 4–6 are the longer runs.
