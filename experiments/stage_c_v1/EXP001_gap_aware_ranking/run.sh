#!/usr/bin/env bash
set -euo pipefail

# EXP001 — run only after human review. Train and Val are used; Test is not.
DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-../../data/ActiveView}"
EXP_NAME="EXP001_gap_aware_ranking"
OUT="${DATA_ROOT}/experiments/stage_c_v1/${EXP_NAME}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

mkdir -p "${OUT}/checkpoints" "${OUT}/runtime"
git -C "${REPO_ROOT}" rev-parse HEAD > "${OUT}/code_commit.txt"
cp "$(dirname "$0")/config.yaml" "${OUT}/config.yaml"

(cd "${REPO_ROOT}" && python -m activeview.scripts.train_stage_c \
  --model-type set_ranker \
  --feature-root "${DATA_ROOT}/datasets/policy_v11_5/stage_c" \
  --stage-b-root "${DATA_ROOT}/datasets/policy_v11_5/stage_b" \
  --output-dir "${OUT}/checkpoints" \
  --batch-size 128 \
  --episodes-per-record 16 \
  --max-epochs 100 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --lambda-reg 1.0 \
  --lambda-rank 1.0 \
  --tau 0.5 \
  --lambda-gap 1.0 \
  --tau-gap 1.0 \
  --max-gap-weight 10.0 \
  --seed 42)

(cd "${REPO_ROOT}" && python -m activeview.scripts.evaluate_stage_c_val \
  --model-type set_ranker \
  --feature-root "${DATA_ROOT}/datasets/policy_v11_5/stage_c" \
  --stage-b-root "${DATA_ROOT}/datasets/policy_v11_5/stage_b" \
  --dataset-root "${DATA_ROOT}/datasets/policy_v11_5" \
  --checkpoint "${OUT}/checkpoints/set_ranker_best.pth" \
  --output-dir "${OUT}/runtime" \
  --baseline "${REPO_ROOT}/experiments/stage_c_v1/${EXP_NAME}/baseline.json" \
  --experiment-id EXP001)
