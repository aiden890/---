# RLinf-MiBoT — Track B deployment

Portable, deployment-ready bundle to port Xiaomi MiBoT into the RLinf training framework
on a **new** NVIDIA GPU server. Everything that can be prepared without the target server
is done and committed; the remaining gates are marked `NEEDS_SERVER` in
[`readiness_report.json`](../readiness_report.json).

Deploying on the new server = clone this repo, fill `.env`, run **one** bootstrap command.

> Constraints honored: RLinf is pinned (commit `bde6c918…`) and treated as an external
> dependency (cloned, never vendored). The Xiaomi checkpoint and RoboCasa assets are
> host-mounted, never baked into the image. No secrets or absolute server paths are
> embedded. The verified `pirl_flow_sde.py` / `flow_policy.py` / `reward.py` are reused by
> import, not re-implemented.

---

## 1. Required server inputs

Provide these when the server is ready (also listed machine-readably in the readiness
report under `required_server_inputs`):

- **SSH host/alias** of the new server.
- **GPU model + count** and **NVIDIA driver version** (driver ≥ 525 for the CUDA 12.1 image).
- **Storage path + quota** for results and caches (≈40 GB+ free: image + checkpoint + results).
- Confirmation that **Docker** and **nvidia-container-toolkit** are installed.
- **Host mount paths**: RoboCasa assets dir, and the Xiaomi checkpoint dir
  (`Xiaomi-Robotics-1-RoboCasa365`, sha `3a6d0293…`).
- Whether **outbound network** is allowed (needed once to clone RLinf + pull the base
  image; otherwise pre-seed `vendor/rlinf` and the base image offline).

Fill them into `.env`:

```bash
cd integrations/rlinf_mibot
cp .env.example .env
$EDITOR .env      # set RLINF_ASSETS / RLINF_CHECKPOINT / RLINF_RESULTS / RLINF_CACHE /
                  # RLINF_GPUS / RLINF_CUDA_ARCH (from preflight) / RLINF_TRAINER_PORT
```

## 2. First deployment command

One command runs preflight → pin RLinf → build image → validate mounts → smoke. It
**stops before any training**:

```bash
bash bootstrap.sh
```

Expected outputs:
- `[PASS]` lines for every preflight gate (driver, docker, toolkit, disk/RAM, ports,
  network, checkpoint + assets mounts).
- `vendor/rlinf` checked out at `bde6c918642abf9a4776cb1d5fabcc5087dfe195`.
- Image `rlinf-mibot:latest` built.
- Checkpoint file hashes verified against `checkpoint.sha256` (`OK`).
- `results/smoke/smoke_report.json` with imports/cuda/egl/robocasa/checkpoint/volume gates
  all `PASS` on the server.
- Final message: *"Deployment is READY up to the non-training gates."*

Run only the host checks first if you prefer: `bash bootstrap.sh preflight`.

## 3. Smoke command

Health/smoke checks alone (no training), any time after the image exists:

```bash
bash run.sh smoke
```

Expected output: `results/smoke/smoke_report.json`, `summary={'PASS': 6, 'FAIL': 0, ...}`
on the GPU server (imports, CUDA tensor, EGL render, RoboCasa reset/step, checkpoint +
processor load with recorded config hash, artifact-volume write).

## 4. PPO smoke command

Tiny **adapter-only** PPO correctness probe — exactly one short update, VLM/base frozen:

```bash
bash run.sh ppo-smoke
```

Expected output: `results/ppo_smoke/ppo_smoke_report.json` with `status: PASS` and one
completed PPO update on the server. GRPO is added only **after** this PPO data flow and the
deterministic eval pass. (Off-server this reports `NEEDS_SERVER`, by design.)

## 5. Resume command

Deterministic ODE eval (`noise_level=0`, bit-exact with the checkpoint ODE) comparing a
saved adapter checkpoint against the base policy on the fixed seeds — this is also the
**resume/eval-a-trained-adapter** path:

```bash
# base policy
bash run.sh ode-eval
# a trained adapter (resume from a saved checkpoint):
bash run.sh ode-eval eval.adapter_checkpoint=/results/<run>/adapter.pt
```

Expected output: `results/ode_eval/ode_eval_report.json` with per-seed deterministic
actions/success and a paired before/after delta.

To resume actual training from a checkpoint (once training is explicitly enabled on the
server), the trainer reads `eval.adapter_checkpoint` / the run dir under `/results`; long
training remains disabled by the config `guards` until explicitly turned on.

## 6. Expected outputs

All under the mounted `RLINF_RESULTS`:
- `smoke/smoke_report.json` — health gates.
- `ppo_smoke/ppo_smoke_report.json` — one PPO update result.
- `ode_eval/ode_eval_report.json` — deterministic eval, per-seed + before/after.
- training runs (when enabled): per-run dir with logs, adapter checkpoint, rollout MP4s.

## 7. RLinf parameter-grid collector (DGX Spark acceptance)

The grid collector is optimizer-free: RLinf `WorkerGroup` actors own independent RoboCasa
simulators while one resident Xiaomi trainer/model server serializes GPU inference and stores
the canonical GRPO rollout chunks. Every expanded config has a stable SHA-256 identity,
disjoint seed band and separate result subtree. Collection exports each trajectory from the
trainer store without updating; `grid-consume` later imports two exported stores and exercises
the existing `op=update -> op=save -> op=load` path exactly once.

This code is not production-ready merely because CPU/fake tests pass. Acceptance requires an
SSH-accessible DGX Spark and saved evidence for all commands below.

```bash
# DGX Spark is aarch64: use the Spark administrator's validated NGC PyTorch image.
# Do not guess this tag and do not use the x86 CUDA 12.1 wheel path.
export RLINF_BASE_IMAGE=<validated-dgx-spark-ngc-pytorch-image>
export RLINF_INSTALL_TORCH=0
export RLINF_INSTALL_FLASH_ATTN=0

bash preflight.sh
bash run.sh build

# Serial baseline and 2-worker parallel run use identical configs/seeds under separate IDs.
bash run.sh model-start dgx-grid-serial
bash run.sh grid-serial dgx-grid-serial
bash run.sh model-stop dgx-grid-serial

bash run.sh model-start dgx-grid-parallel
bash run.sh grid-parallel dgx-grid-parallel validation.inject_fail_once_seed=710000
bash run.sh grid-consume dgx-grid-parallel
bash run.sh model-stop dgx-grid-parallel

bash run.sh grid-compare dgx-grid-serial dgx-grid-parallel
```

`grid-smoke RUN_ID` is the one-command model-start -> parallel collect -> one-update consume ->
cleanup path. `grid-stop RUN_ID` stops both named containers. After an interrupted collector,
first verify no collector container is live, then run `grid-resume RUN_ID`; it preserves a prior
FAILED sentinel as `FAILED.pre-resume.<timestamp>`, clears only the stale lock, and skips job IDs
already present in `episodes.jsonl`.

Required acceptance evidence:

- preflight PASS for hardware, container GPU, EGL, checkpoint, assets and writable results;
- trainer log reaches `GRPO trainer server on ...` after loading the real Xiaomi checkpoint;
- at least two config IDs and two distinct RLinf workers, with
  `audit.cross_config_overlap_seconds > 0`, no mixed config hashes and no duplicate seeds;
- planned injected failure appears once in `failures.jsonl`, then succeeds on bounded retry;
- an interrupted run resumes without duplicating a completed job;
- `consume_smoke/consume_smoke.json` has one real update, adapter delta > 0, no non-finite or
  dropped ratios, and save/load success for the produced checkpoint;
- `serial_vs_parallel.json` reports measured elapsed time, throughput, trainer peak allocation
  and failure attempts. No speedup is claimed until this file exists from real Spark runs.

Future two-Spark placement is configured by `runtime.node_ranks` (for example `[0,0,1,1]`) and
the `runtime.cluster`/`.env` network hooks. The inter-host fabric is **ConnectX-7/RoCE**, not
NVLink. Validate Ray membership, the RoCE interface/GID and shared result/checkpoint mounts on
both nodes before changing `num_nodes`; the single-Spark acceptance does not prove this mode.

## 8. Rollback / removal command

Remove the image and (optionally) the pinned RLinf clone; host mounts (checkpoint, assets,
results) are untouched:

```bash
# stop any running trainer container, drop the image, drop the RLinf clone
docker compose --env-file .env down --remove-orphans 2>/dev/null || true
docker rmi "${RLINF_IMAGE:-rlinf-mibot:latest}" 2>/dev/null || true
rm -rf vendor/rlinf
# results/caches are host-mounted and preserved; delete them yourself if desired.
```

To fully undo the integration, delete `integrations/rlinf_mibot/` and revert the collector
store RPC additions in `rl-train-t_3ed65912/src/{grpo_train_loop,grpo_trainer_server}.py`.

---

## Layout

```
integrations/rlinf_mibot/
  Dockerfile-context:
    docker/Dockerfile              CUDA 12.1.1 + torch 2.5.1 + flash-attn + RLinf@pinned
    configs/requirements-rlinf.*   pinned deps (txt + lock constraint)
    configs/10_nvidia.json         EGL vendor config
  docker-compose.yml / run.sh      server/client-style launch wrappers
  bootstrap.sh / preflight.sh      one-command deploy + read-only host checks
  .env.example                     path/host placeholders only
  REVISIONS.lock / checkpoint.sha256   pinned revisions + checksums
  src/
    smoke.py                       health/smoke gates -> JSON
    mibot_adapter.py               model load / obs transforms / 12-D decode / velocity /
                                   pi-RL sampler (reuses verified rl-env code by import)
    rlinf_env.py                   reward/env adapter + skill contracts + RLinf registration
    train_entry.py / eval_entry.py Hydra PPO-smoke / ODE-eval entry points (guarded)
    grid_collection.py              deterministic grid/seed/artifact contract + audit
    rlinf_grid_runtime.py           production RLinf WorkerGroup scheduler
    rlinf_grid_worker.py            independent RoboCasa simulator worker
    rollout_store.py                portable canonical trainer-store serialization
    consume_collector.py            bounded existing-trainer update/save/load smoke
    compare_grid_runs.py            measured serial-vs-parallel report
  configs/ppo_smoke.yaml           adapter-only PPO smoke (1 iter, long training disabled)
  configs/ode_eval.yaml            deterministic ODE eval (noise=0)
  configs/grid_smoke.yaml          2-config/2-worker collection acceptance config
  tests/test_*.py                  no-GPU unit/static/mutation validation
  readiness_report.json            READY / NEEDS_SERVER per gate
  deploy/README.md                 this file
```

Reused-by-import (single source of truth, not copied):
`rl-env-t_4f3f2b20/src/{pirl_flow_sde,flow_policy,reward,skill_manager}.py`,
`rollouts-xiaomi-t_4a072806/tools/skill_eval.py`, `xiaomi-cu121/rollout.py`.
