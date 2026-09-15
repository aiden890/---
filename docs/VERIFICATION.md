# Verified execution: task t_04bf4fb7

Status: ORIGINAL SIMULATION INFRASTRUCTURE MILESTONE ONLY.
The expanded Xiaomi policy rollout + MP4 goal is NOT complete. A scope-change
comment was first noticed during the final card re-read. See XIAOMI_STATUS.md
for the official inference-runtime blocker and remaining acceptance criteria.

## Environment

- Execution host: v4 (existing SSH connection; not the local aiden-min host)
- Docker: 20.10.18; NVIDIA runtime registered
- GPU: NVIDIA GeForce RTX 3090, 24576 MiB; NVIDIA driver 535.183.01
- Python 3.11.14, RoboCasa 1.0.1, robosuite 1.5.2, MuJoCo 3.3.1,
  NumPy 2.2.5; complete simulation dependency lock: 36 distributions
- Final image: `robocasa-sim:t_04bf4fb7`
- Image ID: `sha256:83c1bc264626107a232003c874cbf7725de4988ddf1e417ac0abfb6aa3a52aba`
- Image size from Docker: 2858344772 bytes
- Asset volume: `robocasa-assets-t_04bf4fb7`, measured `du -sh`: 22G

## Commands actually executed on v4

Working directory: `/home/v4/robocasa-docker-t_04bf4fb7`

```bash
docker build -t robocasa-sim:t_04bf4fb7 .
bash run.sh assets tex fixtures_lw
bash run.sh assets objs_lw objs_objaverse objs_aigen
docker run --rm --gpus all robocasa-sim:t_04bf4fb7 nvidia-smi
bash run.sh smoke
bash run.sh cpu
bash run.sh physics
docker image inspect robocasa-sim:t_04bf4fb7
docker volume inspect robocasa-assets-t_04bf4fb7
```

Final build and all three final simulation commands exited 0. GPU verification
was strengthened with `--require-gpu`, now automatically included by run.sh smoke.

Each final test instantiated RoboCasa Kitchen + PandaOmron, layout/style 1,
seed 0, called reset and executed five steps with 12-dimensional zero actions.
Finite joint state/reward and advancement of physics time were asserted.
Reported simulation time after the test: 0.7500000000000006 seconds, including the
environment's initialization time; it is not the duration of five steps alone.

- GPU EGL: passed, actual GL renderer `NVIDIA GeForce RTX 3090/PCIe/SSE2`;
  128x128 RGB image, pixel standard deviation 78.99487086655054.
- CPU OSMesa, no GPU passthrough: passed, `llvmpipe (LLVM 15.0.6, 256 bits)`.
- Physics-only, no camera rendering or GPU passthrough: passed.
- Both rendered images were nonblank; the GPU frame was visually checked and
  depicts a kitchen, appliances and the Panda robot arm.
- Negative CLI test `--steps 0` rejected the input with exit 2.
- Local `bash -n` and Python compilation passed.
- `python3 verify_results.py` validated all three saved results and exact agreement
  of the 36 locked Python dependencies with the captured `pip freeze`.
- SHA-256 comparison confirmed identical local/remote Dockerfile, requirements,
  smoke test, launch script and NVIDIA EGL descriptor.

## Evidence

- `logs/build-final.log`: final successful Docker build
- `logs/gpu-smoke-final.log`, `logs/cpu-smoke-final.log`,
  `logs/physics-smoke-final.log`: final real execution logs
- `output/smoke/result.json` and `frame.png`: hardware-rendered evidence
- `output/cpu/result.json` and `frame.png`: software-rendered evidence
- `output/physics/result.json`: physics-only evidence
- `logs/pip-freeze.txt`, `logs/versions.txt`, `logs/image-inspect.json`,
  `logs/volume-inspect.json`, `logs/asset-size.txt`: environment manifests
- `logs/assets.log` and `logs/assets-objects.log`: actual downloads

## Resolved failures and limitations

1. Test-first image run failed before image creation. Initial build failed when
   `/opt` exposed checkout directories as Python namespaces; moving to `/app`
   fixed imports without modifying upstream code.
2. Before asset setup, reset failed with missing sink XML; fixture-only setup then
   failed on the scene's knife-block object. Normal object groups were downloaded,
   and the exact scene then passed.
3. Initial EGL rendering silently used Mesa despite successful container
   `nvidia-smi`. Inspection showed NVIDIA driver libraries present but only Mesa's
   EGL vendor JSON. The image now includes `10_nvidia.json`. A real regression
   test asserted NVIDIA rendering, failed on llvmpipe, then passed after the fix.
4. `pip check` reports only omitted `lerobot` and `tianshou`. This is intentional:
   the requested simulation scope uses an explicit simulation-only runtime.
   Training, dataset conversion, optional robots and controllers remain outside
   the supported image. Optional mimicgen/robosuite_models/mink warnings persist.
5. No dataset, VLA policy, training or task-completion evaluation was performed.
   Only Kitchen layout/style 1 with PandaOmron is validated; other environments
   are not implied to have passed. Generative texture downloads were not needed
   or exercised. CPU fallback was tested on v4, not on local aiden-min.
6. Source commits, base digest and Python runtime versions are pinned. Debian
   repositories, build-tool resolution and remote asset archives remain mutable;
   retain the tested image/volume for exact archival reproduction.
7. SSH-agent socket lifetime is external to this project. Existing unrelated
   containers, behavior files and tmux sessions were not changed or stopped.

Durable source and evidence are present locally under
`/home/aiden/Desktop/lab/robot/robocasa-docker` and on v4 under
`/home/v4/robocasa-docker-t_04bf4fb7`. Docker image and assets live on v4's Docker
daemon. The downloadable source/evidence archive intentionally excludes the
multi-gigabyte image, upstream reference clones and asset volume.
