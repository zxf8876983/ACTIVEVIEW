#!/usr/bin/env bash
set -euo pipefail

# EXP022: Train raw executed-candidate utility gate on Train, then evaluate once on Val.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-${REPO_ROOT}/../../data/ActiveView}"
DATASET_ROOT="${DATA_ROOT}/datasets/policy_v11_5"
CACHE_ROOT="${DATASET_ROOT}/stage_d/EXP014_two_step_sequential"
STAGE_B_ROOT="${DATASET_ROOT}/stage_b"
EXP014_ROOT="${DATA_ROOT}/experiments/stage_d/EXP014_two_step_sequential"
EXP017_ROOT="${DATA_ROOT}/experiments/stage_d/EXP017_second_step_gate_calibration"
EXP019_RESULT="${REPO_ROOT}/experiments/stage_d/EXP019_executed_candidate_gate/result.json"
EXP020_RESULT="${REPO_ROOT}/experiments/stage_d/EXP020_contextual_latent_gate/result.json"
OUT="${DATA_ROOT}/experiments/stage_d/EXP022_executed_utility_gate"
EXP014_CHECKPOINT="${EXP014_ROOT}/checkpoints/sequential_observation_ranker_best.pth"
TRAIN_PREDICTIONS="${EXP017_ROOT}/runtime/train_second_step_predictions.jsonl"
VAL_PREDICTIONS="${EXP017_ROOT}/runtime/val_second_step_predictions.jsonl"
V0_PREDICTIONS="${EXP014_ROOT}/v0_predictions/val_predictions.jsonl"
LABEL_MAPPING="${DATA_ROOT}/datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/label_mapping.json"

test -f "${EXP014_CHECKPOINT}"
test -f "${TRAIN_PREDICTIONS}"
test -f "${VAL_PREDICTIONS}"
test -f "${EXP019_RESULT}"
test -f "${EXP020_RESULT}"
mkdir -p "${OUT}"

(cd "${REPO_ROOT}" && python -m activeview.scripts.train_stage_d_utility_gate \
  --cache-root "${CACHE_ROOT}" \
  --stage-b-root "${STAGE_B_ROOT}" \
  --exp014-checkpoint "${EXP014_CHECKPOINT}" \
  --train-predictions "${TRAIN_PREDICTIONS}" \
  --val-predictions "${VAL_PREDICTIONS}" \
  --v0-predictions "${V0_PREDICTIONS}" \
  --exp019-result "${EXP019_RESULT}" \
  --exp020-result "${EXP020_RESULT}" \
  --label-mapping "${LABEL_MAPPING}" \
  --output "${SCRIPT_DIR}/result.json" \
  --runtime-dir "${OUT}" \
  --seed 42 \
  --device cuda:0)
