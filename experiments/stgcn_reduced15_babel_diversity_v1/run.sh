#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/zxf/anaconda3/envs/habitat/bin/python
DATA_ROOT=/home/zxf/WorkSpace/code/data/ActiveView/datasets/reduced15_babel_diversity_v1
STGCN_ROOT="$DATA_ROOT/stgcn_development"
CHECKPOINT=/home/zxf/WorkSpace/code/data/ActiveView/checkpoints/stgcn_reduced15_babel_diversity_v1/stgcn_reduced15_best.pth

"$PYTHON" activeview/scripts/data/prepare_reduced15_dataset.py
"$PYTHON" activeview/scripts/data/generate_reduced15_skeleton_dataset.py \
  --subset stgcn_development --dataset-root "$DATA_ROOT" \
  --split train --split val --device cuda:0
"$PYTHON" activeview/scripts/train/train_selected16_habitat_stgcn.py \
  --data-root "$STGCN_ROOT" --checkpoint "$CHECKPOINT" \
  --device cuda:0 --seed 42 --max-epochs 200 --patience 20 \
  --batch-size 64 --learning-rate 1e-3 --weight-decay 1e-4 \
  --oversample-power 0.5
