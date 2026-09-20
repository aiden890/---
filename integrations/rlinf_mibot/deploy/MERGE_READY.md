# Async rollout + RLinf-native learner merge state

This branch is the non-destructive integration rehearsal for the two active work streams.

## Captured inputs

- Async/group rollout base: `codex/async-3gpu` at `a4ee456`
- Async dirty-tree snapshot commit: `a2dc064`
- Final async test commit: `1c79b1a`
- RLinf-native MiBoT source commit: lab-desktop `6960cdd`
- Cherry-picked native commit in this history: `5d3e164`
- Final async/native merge commit: `6ee83ca`

The original `codex/async-3gpu` working tree was not modified. Its tracked and untracked
code/evidence was copied into `a2dc064`, then the native commit was applied on top.

## Conflict resolution

Only `integrations/rlinf_mibot/run.sh` conflicted. The resolution retains all commands:

- async/group: `model-start`, benchmarks, grid collect/consume/resume;
- native learner: `native-grpo`;
- shared mounts: `/rl_env`, `/train`, `/work`, `/skill_eval_tools`;
- measured 16 GiB shared-memory setting.

No collector, rollout-store, async-policy, trainer-server, or evidence file was changed by
the native commit. The final async delta was merged without conflicts after its 3090
1/2/4-group benchmark completed.

## Final async test delta

Completed as `1c79b1a` and merged through `6ee83ca`. Do not reapply the old snapshot or
the earlier superseded benchmark commit.

## Native integration defects found by image-level testing

- shallow `/integration/src` import crash fixed;
- missing RLinf legacy `gym==0.23.1` dependency pinned;
- incorrect two-node `torchrun` launcher replaced by Ray head/worker/single-driver flow;
- fresh Ray workers now register MiBoT through the opt-in `sitecustomize.py` hook;
- the same fresh-worker hook installs the custom RoboCasa365 state/action/history override;
- policy now implements RLinf `BasePolicy` directly;
- RLinf raw state is converted through the canonical Xiaomi 14-D transform;
- gripper/control-mode action thresholds now match Xiaomi's gym wrapper;
- generic RLinf eval routing now selects MiBoT's noise-free path.

Real checkpoint tests on DGX Spark verified 36 LoRA wrappers / 72 synchronized trainable
tensors, rollout/recompute max log-prob gap `9.5367431640625e-07`, finite backward gradients,
and a fresh Ray GPU worker constructing the registered model.

The native GRASP environment now wraps each RoboCasa subprocess with the verified stable-grasp
predicate and emits one binary relative-reward event after a 20-step hold. MOVE/PLACE reward
and snapshot-entry semantics remain separate parity gates.

## Merged runtime verification (2026-09-19)

Commit `b57e535` completed a native RLinf actor/env/rollout/update run with two environments
and a 32-step horizon. It exited zero, saved `global_step_1`, and reported rollout 12.02 s,
actor update 5.74 s, and no non-finite values. The sampled pair was outcome-homogeneous, so
its zero advantage/gradient was expected rather than counted as a learning gate.

The heterogeneous path was then verified with a fresh Spark group of eight 208-step GRASP
rollouts. One trajectory reached the 20-step stable-grasp predicate at step 148 and seven
timed out, producing exact binary rewards `{1,0}` and 101 stored flow chunks. The RTX 3090
learner imported the hash-checked payloads and completed a real GRPO update: loss
`0.0792375393`, gradient norm `0.0152587891`, adapter delta L2 `0.0088929655`, epoch-0
ratio `1.0000236382`, non-finite count 0, dropped count 0. The LoRA/optimizer/RNG checkpoint
was saved and strictly reloaded, then adapter version 1/hash
`11de2fdd98733a7f9dea0b99edc5d4c671125fd43948cbc44c6d675976430a63` was loaded back
on Spark with an exact identity match.

This full-size update exposed and fixed a client lifetime timeout: the prior 60-second
connection timeout incorrectly also bounded the GPU update, which took 86.76 seconds.
`c37fbef` retains a 60-second connect timeout and switches the established RPC socket to
blocking mode. `e1d8ddf` adds the long-running coordinator and production consume config;
mixed groups update for two epochs, homogeneous groups are skipped, every accepted update
saves a checkpoint, and actor/learner version+hash equality is checked before continuing.

## Required final checks

```bash
python3 -m py_compile integrations/rlinf_mibot/src/*.py
python3 integrations/rlinf_mibot/tests/test_mibot_history.py
python3 integrations/rlinf_mibot/tests/test_static.py
python3 -m unittest discover -s integrations/rlinf_mibot/tests
bash -n integrations/rlinf_mibot/run.sh
docker compose -f integrations/rlinf_mibot/docker-compose.yml config --quiet
git diff --check
```

Run the Python suite inside the project image: the bare macOS interpreter does not include
PyYAML, NumPy, OmegaConf, or Torch. The only bare-host assertion failure observed during the
rehearsal was the macOS `/var` versus `/private/var` temporary-path alias in an existing
collector test.
