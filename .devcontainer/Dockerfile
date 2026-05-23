# NGC PyTorch container: CUDA 11.8 + cuDNN 8.7 + TensorRT 8.5.1 + PyTorch 1.14 + Python 3.8
# TensorRT is pre-installed for deployment workflows.
# RTX 4070 (sm_89) is natively supported by CUDA 11.8 and TRT 8.5 (Ada Lovelace
# kernels), so engine builds no longer need PTX forward-compat fallback.
# 22.04 (TRT 8.2.4) could not compile engines for sm_89; 22.12 is the first NGC
# tag with TRT 8.5.
FROM nvcr.io/nvidia/pytorch:22.12-py3

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

# numpy is pinned to 1.22.4 to match the NGC PyTorch 22.12 ABI (API 0xf, same
# 1.22.x line as 22.04 — keeping the pin keeps the rest of the stack working).
# mmdet3d 0.17.1 declares numpy<1.20, but PyTorch in this image was built
# against the 1.20+ ABI and fails at tensor.numpy() with "Numpy is not
# available" if older numpy is installed — so we ignore the mmdet3d pin.
# numba is pinned to 0.53.1 (the version shipped by the NGC base image).
# mmdet3d 0.17.1 declares numba==0.48.0, but 0.48 requires numpy<1.18 which
# conflicts with the numpy pin above — so we ignore the mmdet3d pin and
# patch the one stale `numba.errors` import below in the mmdet3d stage.
# networkx is pinned to 2.5: trimesh requires `arbitrary_element` which was
# removed in networkx 3.0, and 2.8.x has a circular import issue here.
# psutil is pinned to 5.9.0 to match the C extension shipped with the base
# image (mismatched versions error with "module 'psutil._psutil_linux' has
# no attribute 'getpagesize'").
# opencv-python is pinned to 4.5.5.64 — last cp36-abi3 wheel built against
# the numpy 1.19 C ABI; newer wheels are still fine with numpy 1.22 here.
# scikit-image is pinned to 0.19.3 — 0.20+ requires networkx>=2.8, which
# conflicts with the networkx==2.5 pin needed by trimesh.
# nuscenes-devkit 1.1.11 ships DetectionConfig.class_names as a dict_keys
# view, which is not picklable; that breaks DataLoader spawn workers in
# multi_gpu_test. We sed-patch the install to wrap it in list().
# pycuda is pinned to 2022.2.2 — the sdist for 2024+ builds against the
# numpy 0x10 C ABI (numpy>=1.23) and fails to import here against the
# numpy 1.22.4 ABI (0xf) with "module compiled against API version 0x10".
# NGC 22.12 dropped conda; Python packages now live in
# /usr/local/lib/python3.8/dist-packages/.
# The NGC base image ships its own cv2 binding under that dist-packages dir,
# which `pip uninstall` does not remove and which the cv2 loader prefers —
# wipe it before installing.
RUN rm -rf /usr/local/lib/python3.8/dist-packages/cv2 \
    && pip install --no-cache-dir \
        "numpy==1.22.4" \
        "numba==0.53.1" \
        "networkx==2.5" \
        "psutil==5.9.0" \
        lyft_dataset_sdk \
        "nuscenes-devkit==1.1.11" \
        plyfile \
        "scikit-image==0.19.3" \
        tensorboard \
        "trimesh>=2.35.39,<2.35.40" \
        "opencv-python==4.5.5.64" \
        einops \
        ipython \
        terminaltables \
        filterpy \
        "pycuda==2022.2.2" \
    && sed -i 's|self.class_names = self.class_range.keys()|self.class_names = list(self.class_range.keys())|' \
        /usr/local/lib/python3.8/dist-packages/nuscenes/eval/detection/data_classes.py

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
        /usr/local/lib/python3.8/dist-packages/mmseg/__init__.py

# mmdetection3d
# v0.17.1 was written against PyTorch <=1.10 and uses several APIs that were
# removed/renamed in 1.11+. We migrate them in-place before building:
#   - <THC/THC.h>             -> <ATen/cuda/CUDAContext.h>  (header replaced, not deleted, so at::cuda::getCurrentCUDAStream resolves)
#   - extern THCState *state; -> deleted (THCState no longer exists; nothing in mmdet3d ops actually dereferences it)
#   - THCudaCheck             -> AT_CUDA_CHECK              (drop-in CUDA error check)
#   - THCCeilDiv              -> at::ceil_div               (drop-in)
#   - AT_CHECK                -> TORCH_CHECK                (renamed in 1.5)
# Three more in-place patches on top:
#   - bump mmcv_maximum_version 1.4.0 -> 1.7.0 (same reason as mmseg above)
#   - add force=True to spconv CONV_LAYERS.register_module() — mmcv >=1.5
#     ships its own SparseConv2d/3d under the same registry name, so the
#     mmdet3d registrations otherwise raise "already registered". Forcing
#     override picks mmdet3d's variants, which PETR's configs reference.
#   - rewrite `from numba.errors` -> `from numba.core.errors`. The old
#     module path was removed in numba 0.49; we run 0.53.1 (see numba pin).
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
    && sed -i 's|from numba.errors import|from numba.core.errors import|' \
        mmdet3d/datasets/pipelines/data_augment_utils.py \
    && pip install --no-cache-dir -r requirements/build.txt \
    && python3 setup.py develop --no-deps

WORKDIR /workspace/PETR

CMD ["/bin/bash"]
