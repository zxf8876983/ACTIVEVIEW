#!/usr/bin/env bash
set -euo pipefail

# EXP013 — frozen Stage C-v0 Val Top-K audit; no training and no Test.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-${REPO_ROOT}/../../data/ActiveView}"
PREDICTIONS="${DATA_ROOT}/datasets/policy_v11_5/stage_c/evaluations/predictions/set_ranker_val.jsonl"
STAGE_B_UTILITY="${DATA_ROOT}/datasets/policy_v11_5/stage_b/utility_labels/val.jsonl"
OUT="${DATA_ROOT}/experiments/stage_c_v3/EXP013_topk_reachability/topk_reachability.json"

mkdir -p "$(dirname "${OUT}")"
(cd "${REPO_ROOT}" && python -m activeview.scripts.analyze_stage_c_v3_topk \
  --predictions "${PREDICTIONS}" --stage-b-utility "${STAGE_B_UTILITY}" \
  --output "${OUT}")
