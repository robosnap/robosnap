FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PY_GUI=3.12
ARG PY_ASSET=3.11
ARG PY_ARTICULATE=3.10

ENV ROBOSNAP_ROOT=/workspace/robosnap \
    CHECKPOINT_DIR=/workspace/robosnap/checkpoints \
    HF_HOME=/workspace/robosnap/checkpoints/hf_cache \
    TORCH_HOME=/workspace/robosnap/checkpoints/torch_cache \
    PY_SAM3=/opt/conda/envs/robosnap-gui/bin/python \
    PY_ASSET=/opt/conda/envs/robosnap-asset/bin/python \
    PY_ARTICULATE=/opt/conda/envs/robosnap-articulate/bin/python \
    PORT=7897 \
    SHARE=0 \
    PUBLIC_DEMO=0 \
    ROBOSNAP_KEEP_PROXY=0

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    wget \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -o /tmp/miniforge.sh \
 && bash /tmp/miniforge.sh -b -p /opt/conda \
 && rm /tmp/miniforge.sh \
 && /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/bin:$PATH

RUN mamba create -y -n robosnap-gui python=${PY_GUI} \
 && mamba create -y -n robosnap-asset python=${PY_ASSET} \
 && mamba create -y -n robosnap-articulate python=${PY_ARTICULATE} \
 && mamba clean -afy

WORKDIR /workspace/robosnap
COPY . /workspace/robosnap

RUN mamba run -n robosnap-gui python -m pip install --upgrade pip setuptools wheel \
 && mamba run -n robosnap-gui python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision \
 && mamba run -n robosnap-gui python -m pip install -e . \
 && mamba run -n robosnap-gui python -m pip install -e third_party/sam3

RUN mamba run -n robosnap-asset python -m pip install --upgrade pip setuptools wheel packaging ninja \
 && mamba run -n robosnap-asset python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision \
 && mamba run -n robosnap-asset python -m pip install -e "third_party/sam-3d-objects[inference,p3d]"

RUN mamba run -n robosnap-articulate python -m pip install --upgrade pip setuptools wheel ninja \
 && mamba run -n robosnap-articulate python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision \
 && mamba run -n robosnap-articulate python -m pip install viser fpsample trimesh numba gradio scikit-learn websockets opencv-python pillow numpy \
 && cd third_party/Hunyuan3D-Part/P3-SAM/utils/chamfer3D \
 && mamba run -n robosnap-articulate python setup.py install

RUN mkdir -p /workspace/robosnap/checkpoints /workspace/robosnap/outputs \
 && chmod +x /workspace/robosnap/scripts/run_gui.sh /workspace/robosnap/scripts/gui/bash/*.sh

COPY docker/entrypoint.sh /usr/local/bin/robosnap-entrypoint
RUN chmod +x /usr/local/bin/robosnap-entrypoint

EXPOSE 7897
ENTRYPOINT ["/usr/local/bin/robosnap-entrypoint"]
CMD ["bash", "scripts/run_gui.sh"]
