FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    NUMBA_CACHE_DIR=/tmp/numba
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libegl1 libgl1 libgles2 libglvnd0 libosmesa6 \
    libglib2.0-0 libsm6 libxext6 libxrender1 libx11-6 libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt
ARG ROBOCASA_COMMIT=4f8a2980def75a55dff96b990745b83540425f09
ARG ROBOSUITE_COMMIT=5ce6643f3092639d08f7b0f90ed1c6a84f50552c
RUN git clone https://github.com/robocasa/robocasa.git && cd robocasa && git checkout "$ROBOCASA_COMMIT"
RUN git clone https://github.com/ARISE-Initiative/robosuite.git && cd robosuite && git checkout "$ROBOSUITE_COMMIT"
COPY configs/requirements-sim.txt configs/requirements.lock /app/
RUN pip install -r /app/requirements-sim.txt -c /app/requirements.lock && \
    pip install --no-deps -e /opt/robosuite -e /opt/robocasa
WORKDIR /app
RUN python -m robocasa.scripts.setup_macros
COPY configs/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY scripts/smoke_test.py /app/smoke_test.py
WORKDIR /app
CMD ["python", "/app/smoke_test.py", "--render"]
