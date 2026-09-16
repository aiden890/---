# RLinf-MiBoT GPU training/eval image.
# Follows the verified xiaomi-cu121 server image style (same CUDA base + digest,
# same torch/transformers/flash-attn pins) and layers RLinf (pinned commit) on top.
#
# The Xiaomi checkpoint, RoboCasa assets, results and caches are HOST-MOUNTED at
# run time (see run.sh / docker-compose.yml / .env.example) — never baked in.
# RLinf is CLONED at a pinned commit and installed with --no-deps so our lock file
# stays the single source of truth for versions.
#
# Build args expose the only host-selectable knobs (GPU arch, RLinf commit, jobs).
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04@sha256:7012e535a47883527d402da998384c30b936140c05e2537158c80b8143ee7425

# --- host-selectable build args (defaults suit RTX 3090 / A-series, sm_86) -----------
ARG TORCH_CUDA_ARCH_LIST=8.6
ARG MAX_JOBS=4
ARG RLINF_COMMIT=bde6c918642abf9a4776cb1d5fabcc5087dfe195
ARG RLINF_REPO=https://github.com/RLinf/RLinf.git

ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 \
    MAX_JOBS=${MAX_JOBS} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    NUMBA_CACHE_DIR=/tmp/numba

# EGL + build toolchain (libegl/libgl/libgles for RoboCasa render smoke inside this image).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git ca-certificates \
    libegl1 libgl1 libgles2 libglvnd0 libosmesa6 libglib2.0-0 libsm6 libxext6 libxrender1 libx11-6 \
    build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pinned build tooling then torch (identical to the verified inference image).
RUN python3 -m pip install pip==25.2 setuptools==80.9.0 wheel==0.45.1 packaging==25.0 ninja==1.13.0
RUN python3 -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Locked application deps (transformers/accelerate/etc + RLinf runtime deps). Constraint
# file keeps transitive versions frozen.
COPY configs/requirements-rlinf.txt configs/requirements-rlinf.lock /app/
RUN python3 -m pip install -r /app/requirements-rlinf.txt -c /app/requirements-rlinf.lock
RUN python3 -m pip install flash-attn==2.8.3 --no-build-isolation -c /app/requirements-rlinf.lock

# RLinf at the pinned commit, installed editable & WITHOUT deps (our lock is authoritative).
# Treated as an external dependency: cloned, not vendored into this repo.
RUN git clone "${RLINF_REPO}" /opt/rlinf && cd /opt/rlinf && git checkout "${RLINF_COMMIT}" && \
    (python3 -m pip install --no-deps -e /opt/rlinf || echo "RLINF editable install non-fatal; PYTHONPATH fallback used")
ENV PYTHONPATH=/opt/rlinf:/integration/src:/rl_env/src:/skill_eval_tools

# EGL vendor config for NVIDIA rendering (copied from parent sim image config).
COPY configs/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Record exact resolved env for provenance/audit.
RUN python3 -m pip freeze > /app/requirements.resolved.txt

# The integration layer is mounted read-only at /integration at run time (not baked),
# so a code edit needs no rebuild. Default command prints the smoke help.
CMD ["python3", "-c", "print('rlinf-mibot image ready; run smoke.py --help via run.sh')"]
