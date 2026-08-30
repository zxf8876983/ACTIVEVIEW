#!/usr/bin/env bash
set -euo pipefail

# EXP018 is a Val-only offline diagnostic. It never reads Test or runs training.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-${REPO_ROOT}/../../data/ActiveView}"
DATASET_ROOT="${DATA_ROOT}/datasets/policy_v11_5"
CACHE_ROOT="${DATASET_ROOT}/stage_d/EXP014_two_step_sequential"
STAGE_B_ROOT="${DATASET_ROOT}/stage_b"
EXP014_ROOT="${DATA_ROOT}/experiments/stage_d/EXP014_two_step_sequential"
EXP017_ROOT="${DATA_ROOT}/experiments/stage_d/EXP017_second_step_gate_calibration"
OUT="${DATA_ROOT}/experiments/stage_d/EXP018_executed_candidate_gate_alignment"
V0_PREDICTIONS="${EXP014_ROOT}/v0_predictions/val_predictions.jsonl"
EXP014_PREDICTIONS="${EXP014_ROOT}/runtime/val_second_step_predictions.jsonl"
LABEL_MAPPING="${DATA_ROOT}/datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/label_mapping.json"

test -f "${EXP014_PREDICTIONS}"
test -f "${EXP017_ROOT}/calibration.json"
test -f "${EXP017_ROOT}/result.json"
mkdir -p "${OUT}"
(cd "${REPO_ROOT}" && python -m activeview.scripts.analyze_stage_d_executed_gate_alignment \
  --cache-root "${CACHE_ROOT}" \
  --stage-b-root "${STAGE_B_ROOT}" \
  --exp014-predictions "${EXP014_PREDICTIONS}" \
  --exp017-calibration "${EXP017_ROOT}/calibration.json" \
  --exp017-result "${EXP017_ROOT}/result.json" \
  --v0-predictions "${V0_PREDICTIONS}" \
  --label-mapping "${LABEL_MAPPING}" \
  --output "${SCRIPT_DIR}/result.json" \
  --runtime-output "${OUT}/result.json" \
  --split val)
