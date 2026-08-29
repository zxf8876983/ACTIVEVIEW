#!/usr/bin/env bash
set -euo pipefail

# EXP012 — Train-reference/Val-query audit; no model training and no Test.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-${REPO_ROOT}/../../data/ActiveView}"
FEATURE_ROOT="${DATA_ROOT}/datasets/policy_v11_5/stage_c"
STAGE_B_ROOT="${DATA_ROOT}/datasets/policy_v11_5/stage_b"
PREDICTIONS="${FEATURE_ROOT}/evaluations/predictions/set_ranker_val.jsonl"
LABEL_MAPPING="${DATA_ROOT}/datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/label_mapping.json"
OUT="${DATA_ROOT}/experiments/stage_c_v3/EXP012_predictability_audit/predictability_audit.json"

mkdir -p "$(dirname "${OUT}")"
(cd "${REPO_ROOT}" && python -m activeview.scripts.analyze_stage_c_v3_predictability \
  --feature-root "${FEATURE_ROOT}" --stage-b-root "${STAGE_B_ROOT}" \
  --v0-predictions "${PREDICTIONS}" --label-mapping "${LABEL_MAPPING}" \
  --output "${OUT}")
