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
# opencv-python is pinned to 4.5.5.64 — last cp36-abi3 wheel built against the
# numpy 1.19 C ABI (API 0xd). Newer wheels (>=4.6) require numpy>=1.20 ABI and
# fail at import with "module compiled against API version 0xf but this version
# of numpy is 0xd". The NGC base image also ships its own cv2 binding under
# /opt/conda/lib/python3.8/site-packages/cv2/python-3.8/, which `pip uninstall`
# does not remove and which the cv2 loader prefers — wipe it before installing.
RUN rm -rf /opt/conda/lib/python3.8/site-packages/cv2 \
    && pip install --no-cache-dir \
        "numpy<1.20.0" \
        "numba==0.48.0" \
        "networkx>=2.2,<2.3" \
        lyft_dataset_sdk \
        nuscenes-devkit \
        plyfile \
        scikit-image \
        tensorboard \
        "trimesh>=2.35.39,<2.35.40" \
        "opencv-python==4.5.5.64" \
        ipython \
    || true

# --- MM-series stack (matches install.md, versions adjusted for torch 1.12 + cu116) ---
ARG MMCV_VERSION=1.6.0
ARG MMDET_VERSION=v2.24.1
ARG MMSEG_VERSION=0.20.2
ARG MMDET3D_VERSION=v0.17.1

# mmcv-full: built from source (NOT prebuilt wheel).
# OpenMMLab's prebuilt wheels at download.openmmlab.com are compiled against
# vanilla PyTorch (CXX11 ABI=0). NGC PyTorch is built with CXX11 ABI=1 and
# merges libtorch_cuda_{cu,cpp}.so into a single libtorch_cuda.so, so the
# prebuilt wheel fails to load with either "undefined symbol: ...Ss" (ABI
# mismatch) or "libtorch_cuda_cu.so: cannot open shared object file" (lib
# layout mismatch). Source build picks up this container's PyTorch ABI.
# MAX_JOBS=4 avoids OOM when ninja parallelises 32 nvcc invocations at once.
RUN git clone --depth=1 --branch v${MMCV_VERSION} https://github.com/open-mmlab/mmcv.git /workspace/mmcv \
    && cd /workspace/mmcv \
    && pip install --no-cache-dir -r requirements/runtime.txt \
    && MAX_JOBS=4 pip install --no-cache-dir --no-build-isolation -e .

# mmdetection
RUN git clone https://github.com/open-mmlab/mmdetection.git /workspace/mmdetection \
    && cd /workspace/mmdetection \
    && git checkout ${MMDET_VERSION} \
    && pip install --no-cache-dir -r requirements/build.txt \
    && python3 setup.py develop --no-deps

# mmsegmentation
# 0.20.2 hard-caps mmcv at 1.5.0 in __init__.py; relax to 1.7.0 since mmcv 1.6.x
# is the only line with cu116/torch1.12.0 support and works fine here in practice.
RUN pip install --no-cache-dir "mmsegmentation==${MMSEG_VERSION}" \
    && sed -i "s/^MMCV_MAX = '1.5.0'/MMCV_MAX = '1.7.0'/" \
        /opt/conda/lib/python3.8/site-packages/mmseg/__init__.py

# mmdetection3d
# v0.17.1 was written against PyTorch <=1.10 and uses several APIs that were
# removed/renamed in 1.11+. We migrate them in-place before building:
#   - <THC/THC.h>             -> <ATen/cuda/CUDAContext.h>  (header replaced, not deleted, so at::cuda::getCurrentCUDAStream resolves)
#   - extern THCState *state; -> deleted (THCState no longer exists; nothing in mmdet3d ops actually dereferences it)
#   - THCudaCheck             -> AT_CUDA_CHECK              (drop-in CUDA error check)
#   - THCCeilDiv              -> at::ceil_div               (drop-in)
#   - AT_CHECK                -> TORCH_CHECK                (renamed in 1.5)
# Two more in-place patches on top:
#   - bump mmcv_maximum_version 1.4.0 -> 1.7.0 (same reason as mmseg above)
#   - add force=True to spconv CONV_LAYERS.register_module() — mmcv >=1.5
#     ships its own SparseConv2d/3d under the same registry name, so the
#     mmdet3d registrations otherwise raise "already registered". Forcing
#     override picks mmdet3d's variants, which PETR's configs reference.
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
    && sed -i "s/^mmcv_maximum_version = '1.4.0'/mmcv_maximum_version = '1.7.0'/" mmdet3d/__init__.py \
    && sed -i 's/@CONV_LAYERS.register_module()$/@CONV_LAYERS.register_module(force=True)/' \
        mmdet3d/ops/spconv/conv.py \
    && pip install --no-cache-dir -r requirements/build.txt \
    && python3 setup.py develop --no-deps

WORKDIR /workspace/PETR

CMD ["/bin/bash"]
