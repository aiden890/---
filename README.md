# RoboCasa simulation container

Expanded task status: NOT COMPLETE. Xiaomi-Robotics-1 deployment and a real policy
rollout MP4 were added to the goal during execution. See XIAOMI_STATUS.md for the
CUDA-12.8 host-driver blocker. This README covers only the verified simulator
milestone; no policy episode/video is claimed.

Scope: simulation infrastructure for skill-conditioned VLA research. No VLA model,
training stack, policy checkpoints, or demonstration datasets are included.

## Locations and versions

- Durable local source: `/home/aiden/Desktop/lab/robot/robocasa-docker`
- Execution host: `v4`; remote source: `/home/v4/robocasa-docker-t_04bf4fb7`
- Docker image: `robocasa-sim:t_04bf4fb7`
- Python 3.11.14; RoboCasa 1.0.1; robosuite 1.5.2; MuJoCo 3.3.1; NumPy 2.2.5
- RoboCasa Git revision: `4f8a2980def75a55dff96b990745b83540425f09`
- robosuite Git revision: `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`
- Base image digest and Git commits are pinned in Dockerfile. Python runtime
  dependencies are pinned in `requirements.lock`. Debian security packages and
  Python build tooling can change on rebuild; this is not a bit-identical or
  offline build. Preserve the tested image with `docker image save` if required.

Existing `behavior/` and the `robocasa challenge` tmux session were not modified.
The root `robot/` directory is not a Git repository. `vendor/` contains upstream
reference checkouts, excluded from the build; the Dockerfile checks out its own
pinned sources and therefore needs GitHub, Debian and PyPI network access.

## Build and assets

Run on the machine whose Docker daemon will execute the simulation:

```bash
cd /home/v4/robocasa-docker-t_04bf4fb7
bash run.sh build
bash run.sh assets tex fixtures_lw objs_lw objs_objaverse objs_aigen
```

Asset download is separate from image build. The official downloader prints a
~10 GB estimate; the verified normal asset groups occupy about 22 GiB extracted.
Leave at least 40 GiB free for assets, intermediate ZIPs and the image. It downloads
from the Box links embedded in the pinned RoboCasa repo.
It can return zero even after download errors: inspect its log and require a
passing smoke test, not just a zero exit code.

Named Docker volume: `robocasa-assets-t_04bf4fb7`
Container mount: `/opt/robocasa/robocasa/models/assets`

Docker initializes a fresh named volume with the metadata already present in the
image. Do not replace it with an empty bind mount: that hides the scene/registry
files. A Docker volume is local to its daemon, not shared between local and v4.
Find its host path with:

```bash
docker volume inspect robocasa-assets-t_04bf4fb7
```

Textures, Lightwheel fixtures, Lightwheel objects, Objaverse objects and AI-created
objects support normal scenes. Kitchen scenes need object assets even without a
pick-and-place task (e.g. the fixed knife block). Generative textures are optional:

```bash
bash run.sh assets tex_generative
# Or download all groups:
bash run.sh assets all
```

Downloads are not a per-file checksum-locked artifact. Re-running the official
downloader can overwrite assets. Do not download into a volume currently used by
another rollout. Do not remove the volume when removing a container.
No demonstration dataset or model checkpoint is needed for this smoke test.

## Headless GPU rendering

Host requirements: Docker, NVIDIA driver, NVIDIA Container Toolkit and a working
`--gpus all` runtime. v4 has an RTX 3090 (24 GB), driver 535.183.01 and Docker
20.10.18. The container uses the host's driver through the toolkit, not a driver
installed inside the image. No display server, X11 forwarding, privileged mode,
or host networking is required.

```bash
docker run --rm --gpus all robocasa-sim:t_04bf4fb7 nvidia-smi
bash run.sh smoke
```

Equivalent explicit command:

```bash
mkdir -p output/smoke
docker run --rm --gpus all --shm-size=2g \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v robocasa-assets-t_04bf4fb7:/opt/robocasa/robocasa/models/assets \
  -v "$PWD/output/smoke:/output" \
  robocasa-sim:t_04bf4fb7 python /app/smoke_test.py --render --require-gpu
```

The NVIDIA EGL ICD descriptor `10_nvidia.json` is included explicitly because
v4's runtime mounts the driver libraries but does not supply that descriptor.
`--require-gpu` asserts an NVIDIA OpenGL renderer: `nvidia-smi` alone does not
prove hardware rendering, and EGL can otherwise silently fall back to llvmpipe.

The test creates `Kitchen` with `PandaOmron`, layout 1/style 1/seed 0, calls reset,
executes five zero-action steps, checks finite physics and advancing simulation
time, then validates a nonblank 128x128 RGB frame. This validates simulation, not
completion of a robot task. JSON and PNG outputs are in `output/smoke/`.

## CPU fallback and re-runs

```bash
# No GPU passthrough; OSMesa software offscreen rendering:
bash run.sh cpu
# Headless physics only, camera rendering disabled:
bash run.sh physics
# Longer run, or a different registered task (needs appropriate assets):
bash run.sh smoke --steps 20
bash run.sh smoke --env OpenCabinet
```

`output/cpu` and `output/physics` keep their results separate. Containers are
removed after each run; the asset volume persists. Output files can be root-owned
because the container runs as root. Override `ROBOCASA_IMAGE` and
`ROBOCASA_ASSET_VOLUME` environment variables to isolate another experiment.

## Dependency boundary and troubleshooting

This is deliberately a simulation-only install: `pip install --no-deps -e` uses
the explicit runtime set rather than RoboCasa's training/conversion extras.
`lerobot`, `tianshou`, mimicgen, robosuite_models and mink are not installed.
Consequently `pip check` may report the omitted declared RoboCasa dependencies;
this image does not claim to support training, conversion, optional robots or IK.
Warnings for those optional modules are expected with PandaOmron.

Do not launch Python from `/opt`: editable package namespace directories there
can shadow actual packages (`robosuite.__version__` missing). Use `/app`, the image
working directory. Set both MUJOCO_GL and PYOPENGL_PLATFORM before Python imports.
If a file under `models/assets` is missing, install its asset group; do not mock
assets or change the smoke test to a robosuite-only environment.

`remote.sh` is only a local convenience wrapper for the existing unlocked SSH
agent. It is not included in the image, contains no keys, and the agent socket may
expire. Normal builds and runs on v4 do not need it. The established SSH host key
is checked strictly; no SSH configuration was changed by this task.

## Verification evidence

See `logs/` for real build/download/failure/smoke logs and `output/` for successful
run artifacts. Final validated results are recorded in `VERIFICATION.md`.

References:
- https://robocasa.ai/docs/introduction/installation.html
- https://github.com/robocasa/robocasa
- https://github.com/ARISE-Initiative/robosuite
