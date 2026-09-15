# GR00T N1.5 RoboCasa365 combined eval image (server+client in one process).
# Built ON TOP of the proven xiaomi-cu121 base (torch 2.5.1 cu121 + flash-attn 2.8.3),
# reusing the flash-attn wheel so we skip the ~30 min compile. We add EGL GPU-render
# libraries, robosuite+robocasa (pinned commits = same as robocasa-sim image), the
# robocasa-benchmark Isaac-GR00T fork (pinned), and the curated GR00T runtime deps.
ARG BASE_IMAGE=xiaomi-cu121:t_9f03a613
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    NUMBA_CACHE_DIR=/tmp/numba HF_HUB_ENABLE_HXF_TRANSFER=0

# --- EGL / OSMesa GPU-render stack (mujoco egl) that the cu121 base lacks ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglvnd0 libglvnd-dev libosmesa6 libgl1-mesa-dri \
    libglib2.0-0 libsm6 libxext6 libxrender1 libx11-6 \
    && rm -rf /var/lib/apt/lists/*
# NVIDIA EGL vendor ICD so mujoco picks the GPU device.
COPY groot/configs/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /opt
ARG ROBOCASA_COMMIT=4f8a2980def75a55dff96b990745b83540425f09
ARG ROBOSUITE_COMMIT=5ce6643f3092639d08f7b0f90ed1c6a84f50552c
ARG GROOT_COMMIT=9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10
RUN git clone https://github.com/robocasa/robocasa.git && cd robocasa && git checkout "$ROBOCASA_COMMIT"
RUN git clone https://github.com/ARISE-Initiative/robosuite.git && cd robosuite && git checkout "$ROBOSUITE_COMMIT"
RUN git clone https://github.com/robocasa-benchmark/Isaac-GR00T.git && cd Isaac-GR00T && git checkout "$GROOT_COMMIT"

# Pin transformers to the fork's tested version (base image ships 4.57.1); flash-attn is
# transformers-version independent so downgrading is safe and matches upstream exactly.
RUN python3 -m pip install transformers==4.51.3

# Simulation deps (robosuite/robocasa), then GR00T runtime deps. Both installed --no-deps
# so we don't drag in tensorflow / pytorch3d / a conflicting torch.
COPY groot/configs/requirements-sim.txt groot/configs/requirements.lock groot/configs/requirements-groot.txt /app/
RUN python3 -m pip install -r /app/requirements-sim.txt -c /app/requirements.lock
RUN python3 -m pip install -r /app/requirements-groot.txt
RUN python3 -m pip install --no-deps -e /opt/robosuite -e /opt/robocasa -e /opt/Isaac-GR00T

# robocasa macros (writes ~/.robosuite / macros) and EGL device selection.
RUN python3 -m robocasa.scripts.setup_macros || true

WORKDIR /app
RUN python3 -m pip freeze > /app/requirements.resolved.txt
CMD ["python3", "-c", "import gr00t, robocasa, robosuite; print('groot eval image ok')"]
