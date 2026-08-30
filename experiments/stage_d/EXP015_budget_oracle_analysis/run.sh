#!/usr/bin/env bash
set -euo pipefail

# EXP015 has no training and reads Val artifacts only. EXP014 must exist.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-${REPO_ROOT}/../../data/ActiveView}"
CACHE_ROOT="${DATA_ROOT}/datasets/policy_v11_5/stage_d/EXP014_two_step_sequential"
STAGE_B_ROOT="${DATA_ROOT}/datasets/policy_v11_5/stage_b"
EXP014_OUT="${DATA_ROOT}/experiments/stage_d/EXP014_two_step_sequential"
V0_PREDICTIONS="${EXP014_OUT}/v0_predictions/val_predictions.jsonl"
EXP014_PREDICTIONS="${EXP014_OUT}/runtime/val_second_step_predictions.jsonl"
LABEL_MAPPING="${DATA_ROOT}/datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/label_mapping.json"
OUT="${DATA_ROOT}/experiments/stage_d/EXP015_budget_oracle_analysis"

test -f "${EXP014_PREDICTIONS}" || { echo "Missing EXP014 Val predictions: ${EXP014_PREDICTIONS}" >&2; exit 1; }
mkdir -p "${OUT}"
(cd "${REPO_ROOT}" && python -m activeview.scripts.analyze_stage_d_budget \
  --cache-root "${CACHE_ROOT}" --stage-b-root "${STAGE_B_ROOT}" \
  --exp014-predictions "${EXP014_PREDICTIONS}" \
  --v0-predictions "${V0_PREDICTIONS}" --label-mapping "${LABEL_MAPPING}" \
  --output "${OUT}/budget_oracle_analysis.json")
