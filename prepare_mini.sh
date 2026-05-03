#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

DATA_ROOT=${DATA_ROOT:-./data/nuscenes}

if [ ! -d "$DATA_ROOT/v1.0-mini" ]; then
    echo "Place nuScenes mini under: $DATA_ROOT"
    echo "  $DATA_ROOT/samples/"
    echo "  $DATA_ROOT/sweeps/"
    echo "  $DATA_ROOT/maps/"
    echo "  $DATA_ROOT/v1.0-mini/"
    exit 1
fi

python tools/create_data.py nuscenes \
    --root-path "$DATA_ROOT" \
    --out-dir "$DATA_ROOT" \
    --extra-tag nuscenes \
    --version v1.0-mini

echo "Done. Generated:"
ls -lh "$DATA_ROOT"/nuscenes_infos_*.pkl
