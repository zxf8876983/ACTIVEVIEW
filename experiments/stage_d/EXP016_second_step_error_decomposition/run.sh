#!/usr/bin/env bash
set -euo pipefail

# EXP016 is Val-only and analysis-only. Run only after explicit human review.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-${REPO_ROOT}/../../data/ActiveView}"
DATASET_ROOT="${DATA_ROOT}/datasets/policy_v11_5"
CACHE_ROOT="${DATASET_ROOT}/stage_d/EXP014_two_step_sequential"
STAGE_B_ROOT="${DATASET_ROOT}/stage_b"
EXP014_ROOT="${DATA_ROOT}/experiments/stage_d/EXP014_two_step_sequential"
V0_PREDICTIONS="${EXP014_ROOT}/v0_predictions/val_predictions.jsonl"
EXP014_PREDICTIONS="${EXP014_ROOT}/runtime/val_second_step_predictions.jsonl"
LABEL_MAPPING="${DATA_ROOT}/datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/label_mapping.json"
OUT="${DATA_ROOT}/experiments/stage_d/EXP016_second_step_error_decomposition"

test -f "${EXP014_PREDICTIONS}"
mkdir -p "${OUT}"
(cd "${REPO_ROOT}" && python -m activeview.scripts.analyze_stage_d_second_step_errors \
  --cache-root "${CACHE_ROOT}" \
  --stage-b-root "${STAGE_B_ROOT}" \
  --exp014-predictions "${EXP014_PREDICTIONS}" \
  --v0-predictions "${V0_PREDICTIONS}" \
  --label-mapping "${LABEL_MAPPING}" \
  --output "${OUT}/exp016_analysis.json" \
  --split val)
