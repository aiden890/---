FROM nvidia/cuda:12.1.1-devel-ubuntu22.04@sha256:7012e535a47883527d402da998384c30b936140c05e2537158c80b8143ee7425
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.6 FLASH_ATTENTION_FORCE_BUILD=TRUE
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev git ca-certificates libegl1 libgl1 libgles2 build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN python3 -m pip install pip==25.2 setuptools==80.9.0 wheel==0.45.1 packaging==25.0 ninja==1.13.0
RUN python3 -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
RUN python3 -m pip install transformers==4.57.1 accelerate==1.10.1 einops==0.8.1 pillow==11.3.0 scipy==1.15.3
RUN python3 -m pip install flash-attn==2.8.3 --no-build-isolation
RUN python3 -m pip install imageio==2.37.0 imageio-ffmpeg==0.6.0
RUN python3 -m pip freeze > /app/requirements.resolved.txt
COPY download.py probe.py /app/
CMD ["python3", "/app/probe.py"]
