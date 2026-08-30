#!/usr/bin/env bash
set -euo pipefail

# EXP017 is intentionally not run as part of repository preparation. Execute
# only after explicit human authorization and code review.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-${REPO_ROOT}/../../data/ActiveView}"
DATASET_ROOT="${DATA_ROOT}/datasets/policy_v11_5"
CACHE_ROOT="${DATASET_ROOT}/stage_d/EXP014_two_step_sequential"
STAGE_B_ROOT="${DATASET_ROOT}/stage_b"
EXP014_ROOT="${DATA_ROOT}/experiments/stage_d/EXP014_two_step_sequential"
OUT="${DATA_ROOT}/experiments/stage_d/EXP017_second_step_gate_calibration"
CHECKPOINT="${EXP014_ROOT}/checkpoints/sequential_observation_ranker_best.pth"
V0_PREDICTIONS="${EXP014_ROOT}/v0_predictions/val_predictions.jsonl"
LABEL_MAPPING="${DATA_ROOT}/datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/label_mapping.json"

mkdir -p "${OUT}/runtime"
(cd "${REPO_ROOT}" && python -m activeview.scripts.analyze_stage_d_gate_calibration \
  --cache-root "${CACHE_ROOT}" \
  --stage-b-root "${STAGE_B_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --v0-predictions "${V0_PREDICTIONS}" \
  --label-mapping "${LABEL_MAPPING}" \
  --output-dir "${OUT}" \
  --calibration-artifact "${OUT}/calibration.json" \
  --result-output "${OUT}/result.json" \
  --device cuda:0 \
  --batch-size 128)
