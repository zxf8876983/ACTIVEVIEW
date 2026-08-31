#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-${REPO_ROOT}/../../data/ActiveView}"
RUNTIME_ROOT="${ACTIVEVIEW_RGB_RUNTIME_ROOT:-/home/zxf/MG08/robot/ActiveView}"
PYTHON_BIN="${ACTIVEVIEW_PYTHON:-python}"

"${PYTHON_BIN}" -m activeview.scripts.train_stage_d_rgbd \
  --cache-root "${DATA_ROOT}/datasets/policy_v11_5/stage_d/EXP014_two_step_sequential" \
  --stage-b-root "${DATA_ROOT}/datasets/policy_v11_5/stage_b" \
  --source-root "${DATA_ROOT}/datasets/offline/hm3d-train" \
  --motion-manifest "${DATA_ROOT}/datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/val.json" \
  --scene-root "${ACTIVEVIEW_HABITAT_DATA_ROOT:-${REPO_ROOT}/../robot/DATA}/hm3d-train" \
  --exp022-result "${REPO_ROOT}/experiments/stage_d/EXP022_executed_utility_gate/result.json" \
  --exp024-result "${REPO_ROOT}/experiments/stage_d/EXP024_dinov2_rgb_context/result.json" \
  --exp025-result "${REPO_ROOT}/experiments/stage_d/EXP025_dinov2_spatial_rgb/result.json" \
  --exp025-cache "${RUNTIME_ROOT}/features/dinov2_vitb14_spatial4x4" \
  --depth-cache "${RUNTIME_ROOT}/features/habitat_depth_spatial4x4" \
  --label-mapping "${DATA_ROOT}/datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/label_mapping.json" \
  --rgb-root "${RUNTIME_ROOT}/datasets/offline/hm3d-train" \
  --output "${SCRIPT_DIR}/result.json" \
  --runtime-dir "${RUNTIME_ROOT}/experiments/stage_d/EXP026_spatial_rgbd_utility_regression" \
  --workers 16 --seed 42 --device cuda:0
