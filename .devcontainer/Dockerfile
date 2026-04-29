# NGC PyTorch container: CUDA 11.6 + cuDNN 8.4 + TensorRT 8.2.4 + PyTorch 1.12 + Python 3.8
# TensorRT is pre-installed for deployment workflows.
# RTX 4070 (sm_89) runs via PTX forward-compatibility on CUDA 11.6.
FROM nvcr.io/nvidia/pytorch:22.04-py3

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="8.9" \
    MMCV_WITH_OPS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        unzip \
        sudo \
        vim \
        tmux \
        ca-certificates \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Pin numpy/numba to versions PETR requirements.txt expects, before MM* install.
# (numpy<1.20 / numba==0.48.0 are required by mmdet3d 0.17.1)
RUN pip install --no-cache-dir \
        "numpy<1.20.0" \
        "numba==0.48.0" \
        "networkx>=2.2,<2.3" \
        lyft_dataset_sdk \
        nuscenes-devkit \
        plyfile \
        scikit-image \
        tensorboard \
        "trimesh>=2.35.39,<2.35.40" \
        opencv-python \
        ipython \
    || true

WORKDIR /workspace/PETR

CMD ["/bin/bash"]
