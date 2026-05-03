#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

CONFIG=${CONFIG:-projects/configs/petr/petr_r50dcn_gridmask_p4_mini.py}
CHECKPOINT=${CHECKPOINT:-ckpts/epoch_24.pth}
GPUS=${GPUS:-1}
OUT_DIR=${OUT_DIR:-work_dirs/pp-nus}

if [ ! -f "$CONFIG" ]; then
    echo "Config not found: $CONFIG"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT"
    exit 1
fi

mkdir -p "$OUT_DIR"

tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS" \
    --out "$OUT_DIR/results_eval.pkl" \
    --format-only \
    --eval-options "jsonfile_prefix=$OUT_DIR/results_eval"
