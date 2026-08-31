#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-${REPO_ROOT}/../../data/ActiveView}"
RGB_ROOT="${ACTIVEVIEW_RGB_ROOT:-/home/zxf/MG08/robot/ActiveView/datasets/offline/hm3d-train}"
RUNTIME_ROOT="${ACTIVEVIEW_RGB_RUNTIME_ROOT:-/home/zxf/MG08/robot/ActiveView}"
PYTHON_BIN="${ACTIVEVIEW_PYTHON:-python}"

"${PYTHON_BIN}" -m activeview.scripts.train_stage_d_rgb_spatial \
  --cache-root "${DATA_ROOT}/datasets/policy_v11_5/stage_d/EXP014_two_step_sequential" \
  --stage-b-root "${DATA_ROOT}/datasets/policy_v11_5/stage_b" \
  --exp014-checkpoint "${DATA_ROOT}/experiments/stage_d/EXP014_two_step_sequential/checkpoints/sequential_observation_ranker_best.pth" \
  --train-predictions "${DATA_ROOT}/experiments/stage_d/EXP017_second_step_gate_calibration/runtime/train_second_step_predictions.jsonl" \
  --val-predictions "${DATA_ROOT}/experiments/stage_d/EXP017_second_step_gate_calibration/runtime/val_second_step_predictions.jsonl" \
  --v0-predictions "${DATA_ROOT}/experiments/stage_d/EXP014_two_step_sequential/v0_predictions/val_predictions.jsonl" \
  --exp022-result "${REPO_ROOT}/experiments/stage_d/EXP022_executed_utility_gate/result.json" \
  --exp024-result "${REPO_ROOT}/experiments/stage_d/EXP024_dinov2_rgb_context/result.json" \
  --label-mapping "${DATA_ROOT}/datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/label_mapping.json" \
  --rgb-root "${RGB_ROOT}" \
  --embedding-cache "${RUNTIME_ROOT}/features/dinov2_vitb14_spatial4x4" \
  --output "${SCRIPT_DIR}/result.json" \
  --runtime-dir "${RUNTIME_ROOT}/experiments/stage_d/EXP025_dinov2_spatial_rgb" \
  --seed 42 \
  --device cuda:0
