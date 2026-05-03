#!/usr/bin/env bash
#
# Install script for PETR, following install.md.
#
# Configurable via environment variables:
#   CU_VERSION      CUDA version tag for mmcv wheel index   (default: cu111)
#   TORCH_VERSION   Torch version tag for mmcv wheel index  (default: torch1.9.0)
#   MMCV_VERSION    mmcv-full version                       (default: 1.4.0)
#   MMDET_VERSION   mmdetection git tag                     (default: v2.24.1)
#   MMSEG_VERSION   mmsegmentation pip version              (default: 0.20.2)
#   MMDET3D_VERSION mmdetection3d git tag                   (default: v0.17.1)
#   WORKDIR         Parent directory for clones             (default: parent of this script)
#   NUSCENES_PATH   Path to nuScenes dataset to symlink     (default: /data/Dataset/nuScenes)
#   PIP             pip command                             (default: sudo pip)
#   PY              python command                          (default: sudo python3)

set -euo pipefail

CU_VERSION="${CU_VERSION:-cu116}"
TORCH_VERSION="${TORCH_VERSION:-torch1.12.0}"
MMCV_VERSION="${MMCV_VERSION:-1.6.0}"
MMDET_VERSION="${MMDET_VERSION:-v2.24.1}"
MMSEG_VERSION="${MMSEG_VERSION:-0.20.2}"
MMDET3D_VERSION="${MMDET3D_VERSION:-v0.17.1}"
NUSCENES_PATH="${NUSCENES_PATH:-/data/Dataset/nuScenes}"
PIP="${PIP:-pip}"
PY="${PY:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="$SCRIPT_DIR"
WORKDIR="${WORKDIR:-$(dirname "$PETR_DIR")}"

echo "==> Working in: $WORKDIR"
echo "==> PETR repo:  $PETR_DIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

if $PY -c "import mmcv, mmcv.ops" >/dev/null 2>&1; then
    echo "==> mmcv-full already installed; skipping."
else
    echo "==> Installing mmcv-full==${MMCV_VERSION} (${CU_VERSION}/${TORCH_VERSION})"
    $PIP install "mmcv-full==${MMCV_VERSION}" \
        -f "https://download.openmmlab.com/mmcv/dist/${CU_VERSION}/${TORCH_VERSION}/index.html"
fi

if $PY -c "import mmdet" >/dev/null 2>&1; then
    echo "==> mmdet already installed; skipping clone/build."
else
    echo "==> Installing MMDetection ${MMDET_VERSION}"
    if [ ! -d "$WORKDIR/mmdetection" ]; then
        git clone https://github.com/open-mmlab/mmdetection.git "$WORKDIR/mmdetection"
    fi
    pushd "$WORKDIR/mmdetection" >/dev/null
    git checkout "$MMDET_VERSION"
    $PIP install -r requirements/build.txt
    $PY setup.py develop --no-deps
    popd >/dev/null
fi

if $PY -c "import mmseg" >/dev/null 2>&1; then
    echo "==> mmseg already installed; skipping."
else
    echo "==> Installing MMSegmentation==${MMSEG_VERSION}"
    $PIP install "mmsegmentation==${MMSEG_VERSION}"
fi

if $PY -c "import mmdet3d" >/dev/null 2>&1; then
    echo "==> mmdet3d already installed; skipping clone/build."
else
    echo "==> Installing MMDetection3D ${MMDET3D_VERSION}"
    if [ ! -d "$WORKDIR/mmdetection3d" ]; then
        git clone https://github.com/open-mmlab/mmdetection3d.git "$WORKDIR/mmdetection3d"
    fi
    pushd "$WORKDIR/mmdetection3d" >/dev/null
    git checkout "$MMDET3D_VERSION"
    $PIP install -r requirements/build.txt
    $PY setup.py develop --no-deps
    popd >/dev/null
fi

echo "==> Setting up PETR directories and symlinks"
cd "$PETR_DIR"
mkdir -p ckpts data

if [ ! -e "$PETR_DIR/mmdetection3d" ]; then
    ln -s "$WORKDIR/mmdetection3d" "$PETR_DIR/mmdetection3d"
    echo "    linked mmdetection3d -> $WORKDIR/mmdetection3d"
fi

if [ ! -e "$PETR_DIR/data/nuscenes" ]; then
    if [ -e "$NUSCENES_PATH" ]; then
        ln -s "$NUSCENES_PATH" "$PETR_DIR/data/nuscenes"
        echo "    linked data/nuscenes -> $NUSCENES_PATH"
    else
        echo "    WARNING: NUSCENES_PATH '$NUSCENES_PATH' does not exist; skipping symlink."
        echo "             Set NUSCENES_PATH and re-run, or create the symlink manually:"
        echo "             ln -s <nuscenes_path> $PETR_DIR/data/nuscenes"
    fi
fi

echo "==> Done."
