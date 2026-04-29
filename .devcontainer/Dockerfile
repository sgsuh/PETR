# NGC PyTorch container: CUDA 11.6 + cuDNN 8.4 + TensorRT 8.2.4 + PyTorch 1.12 + Python 3.8
# TensorRT is pre-installed for deployment workflows.
# RTX 4070 (sm_89) runs via PTX forward-compatibility on CUDA 11.6.
FROM nvcr.io/nvidia/pytorch:22.04-py3

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="8.6+PTX" \
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

# --- MM-series stack (matches install.md, versions adjusted for torch 1.12 + cu116) ---
ARG MMCV_VERSION=1.6.2
ARG MMDET_VERSION=v2.24.1
ARG MMSEG_VERSION=0.20.2
ARG MMDET3D_VERSION=v0.17.1

# mmcv-full (prebuilt wheel for cu116 / torch1.12.0)
RUN pip install --no-cache-dir "mmcv-full==${MMCV_VERSION}" \
        -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html

# mmdetection
RUN git clone https://github.com/open-mmlab/mmdetection.git /workspace/mmdetection \
    && cd /workspace/mmdetection \
    && git checkout ${MMDET_VERSION} \
    && pip install --no-cache-dir -r requirements/build.txt \
    && python3 setup.py develop --no-deps

# mmsegmentation
RUN pip install --no-cache-dir "mmsegmentation==${MMSEG_VERSION}"

# mmdetection3d
# v0.17.1 was written against PyTorch <=1.10 and uses several APIs that were
# removed/renamed in 1.11+. We migrate them in-place before building:
#   - <THC/THC.h>             -> <ATen/cuda/CUDAContext.h>  (header replaced, not deleted, so at::cuda::getCurrentCUDAStream resolves)
#   - extern THCState *state; -> deleted (THCState no longer exists; nothing in mmdet3d ops actually dereferences it)
#   - THCudaCheck             -> AT_CUDA_CHECK              (drop-in CUDA error check)
#   - THCCeilDiv              -> at::ceil_div               (drop-in)
#   - AT_CHECK                -> TORCH_CHECK                (renamed in 1.5)
RUN git clone https://github.com/open-mmlab/mmdetection3d.git /workspace/mmdetection3d \
    && cd /workspace/mmdetection3d \
    && git checkout ${MMDET3D_VERSION} \
    && find mmdet3d/ops -type f \( -name '*.cpp' -o -name '*.cu' -o -name '*.h' -o -name '*.cuh' \) -print0 \
        | xargs -0 sed -i \
            -e 's|#include <THC/THC.h>|#include <ATen/cuda/CUDAContext.h>|g' \
            -e '/extern THCState \*state;/d' \
            -e 's/\bTHCudaCheck\b/AT_CUDA_CHECK/g' \
            -e 's/\bTHCCeilDiv\b/at::ceil_div/g' \
            -e 's/\bAT_CHECK\b/TORCH_CHECK/g' \
    && pip install --no-cache-dir -r requirements/build.txt \
    && python3 setup.py develop --no-deps

WORKDIR /workspace/PETR

CMD ["/bin/bash"]
