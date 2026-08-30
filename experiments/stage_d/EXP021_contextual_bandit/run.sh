#!/usr/bin/env bash
set -euo pipefail

# EXP021: fixed Train-only contextual-bandit optimization, then one Val run.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-${REPO_ROOT}/../../data/ActiveView}"
DATASET_ROOT="${DATA_ROOT}/datasets/policy_v11_5"
CACHE_ROOT="${DATASET_ROOT}/stage_d/EXP014_two_step_sequential"
STAGE_B_ROOT="${DATASET_ROOT}/stage_b"
EXP014_ROOT="${DATA_ROOT}/experiments/stage_d/EXP014_two_step_sequential"
EXP019_ROOT="${REPO_ROOT}/experiments/stage_d/EXP019_executed_candidate_gate"
OUT="${DATA_ROOT}/experiments/stage_d/EXP021_contextual_bandit"
V0_PREDICTIONS="${EXP014_ROOT}/v0_predictions/val_predictions.jsonl"
EXP019_RESULT="${EXP019_ROOT}/result.json"
LABEL_MAPPING="${DATA_ROOT}/datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/label_mapping.json"

test -f "${V0_PREDICTIONS}"
test -f "${EXP019_RESULT}"
mkdir -p "${OUT}"
(cd "${REPO_ROOT}" && python -m activeview.scripts.train_stage_d_contextual_bandit \
  --cache-root "${CACHE_ROOT}" \
  --stage-b-root "${STAGE_B_ROOT}" \
  --v0-predictions "${V0_PREDICTIONS}" \
  --exp019-result "${EXP019_RESULT}" \
  --label-mapping "${LABEL_MAPPING}" \
  --output "${SCRIPT_DIR}/result.json" \
  --runtime-dir "${OUT}" \
  --seed 42 \
  --device cuda:0)
