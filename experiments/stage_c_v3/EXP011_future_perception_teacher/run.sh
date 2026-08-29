#!/usr/bin/env bash
set -euo pipefail

# EXP011 — diagnostic-only future-perception teacher; Train -> Val only.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-${REPO_ROOT}/../../data/ActiveView}"
FEATURE_ROOT="${DATA_ROOT}/datasets/policy_v11_5/stage_c"
STAGE_B_ROOT="${DATA_ROOT}/datasets/policy_v11_5/stage_b"
CACHE_ROOT="${DATA_ROOT}/datasets/policy_v11_5/stage_c_v3/future_perception_teacher"
OUT="${DATA_ROOT}/experiments/stage_c_v3/EXP011_future_perception_teacher"
BASELINE_PREDICTIONS="${FEATURE_ROOT}/evaluations/predictions/set_ranker_val.jsonl"

mkdir -p "${OUT}/checkpoints" "${OUT}/runtime"
git -C "${REPO_ROOT}" rev-parse HEAD > "${OUT}/code_commit.txt"
cp "${SCRIPT_DIR}/config.yaml" "${OUT}/config.yaml"
(cd "${REPO_ROOT}" && python -m activeview.scripts.build_stage_c_v3_teacher_cache \
  --feature-root "${FEATURE_ROOT}" --stage-b-root "${STAGE_B_ROOT}" \
  --output-dir "${CACHE_ROOT}")
(cd "${REPO_ROOT}" && python -m activeview.scripts.train_stage_c_v3_teacher \
  --cache-root "${CACHE_ROOT}" --stage-b-root "${STAGE_B_ROOT}" \
  --output-dir "${OUT}/checkpoints" \
  --device cuda:0 --batch-size 128 --episodes-per-record 16 \
  --max-epochs 100 --patience 10 --lr 0.001 --weight-decay 0.0001 --seed 42)
(cd "${REPO_ROOT}" && python -m activeview.scripts.evaluate_stage_c_v3_teacher \
  --cache-root "${CACHE_ROOT}" --stage-b-root "${STAGE_B_ROOT}" \
  --checkpoint "${OUT}/checkpoints/future_perception_teacher_best.pth" \
  --output-dir "${OUT}/runtime" --baseline-predictions "${BASELINE_PREDICTIONS}" \
  --device cuda:0 --batch-size 128 \
  > "${OUT}/result.json")
