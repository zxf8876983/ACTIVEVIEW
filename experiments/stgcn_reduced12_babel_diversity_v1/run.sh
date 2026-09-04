#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/zxf/anaconda3/envs/habitat/bin/python
ROOT=/home/zxf/WorkSpace/code/data/ActiveView/datasets/reduced12_babel_diversity_v1

"$PYTHON" activeview/scripts/data/prepare_reduced12_dataset.py
"$PYTHON" activeview/scripts/data/generate_reduced12_skeleton_dataset.py --subset stgcn_development --dataset-root "$ROOT" --device cuda:0
"$PYTHON" activeview/scripts/data/generate_reduced12_skeleton_dataset.py --subset activeview --dataset-root "$ROOT" --device cuda:0
"$PYTHON" activeview/scripts/train/train_selected16_habitat_stgcn.py \
  --data-root "$ROOT/stgcn_development" \
  --checkpoint /home/zxf/WorkSpace/code/data/ActiveView/checkpoints/stgcn_reduced12_babel_diversity_v1/stgcn_reduced12_best.pth \
  --device cuda:0 --seed 42
