#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

CONFIG=${CONFIG:-projects/configs/petr/petr_vovnet_gridmask_p4_800x320_export.py}
ONNX=${ONNX:-ckpts/petr-vov-p4-800x320/epoch_24.onnx}
ENGINE=${ENGINE:-ckpts/petr-vov-p4-800x320/epoch_24.engine}
DATA_ROOT=${DATA_ROOT:-data/nuscenes/}
ANN_FILE=${ANN_FILE:-data/nuscenes/nuscenes_infos_val.pkl}
OUT=${OUT:-work_dirs/petr_vov_trt/results_eval.pkl}
TRACK=${TRACK:-0}

if [ ! -f "$CONFIG" ]; then
    echo "Config not found: $CONFIG"
    exit 1
fi

if [ ! -f "$ENGINE" ] && [ ! -f "$ONNX" ]; then
    echo "Neither TRT engine ($ENGINE) nor ONNX ($ONNX) found."
    echo "Run tools/export_onnx.py first."
    exit 1
fi

if [ ! -f "$ANN_FILE" ]; then
    echo "Annotation pkl not found: $ANN_FILE"
    echo "Run tools/create_data.py first (or point ANN_FILE to your nuScenes mini info file)."
    exit 1
fi

TRACK_FLAG=""
if [ "$TRACK" = "1" ] || [ "$TRACK" = "true" ]; then
    TRACK_FLAG="--track"
fi

python tools/test_trt.py \
    --config "$CONFIG" \
    --onnx_path "$ONNX" \
    --trt_path "$ENGINE" \
    --data_root "$DATA_ROOT" \
    --ann_file "$ANN_FILE" \
    --out "$OUT" \
    --eval bbox \
    $TRACK_FLAG \
    "$@"
