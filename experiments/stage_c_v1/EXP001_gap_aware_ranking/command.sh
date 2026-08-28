#!/usr/bin/env bash
set -euo pipefail

# Experiment: EXP001 — planned command; do not run before explicit start approval.
# Stage C-v1 Test evaluation is forbidden during development.

DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-../../data/ActiveView}"
python -m activeview.scripts.train_stage_c \
  --model-type set_ranker \
  --feature-root "${DATA_ROOT}/datasets/policy_v11_5/stage_c" \
  --stage-b-root "${DATA_ROOT}/datasets/policy_v11_5/stage_b" \
  --output-dir "${DATA_ROOT}/experiments/stage_c_v1/EXP001_gap_aware_ranking/checkpoints" \
  --lambda-gap 1.0 \
  --tau-gap 1.0 \
  --max-gap-weight 10.0
