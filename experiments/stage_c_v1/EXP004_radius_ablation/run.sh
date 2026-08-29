#!/usr/bin/env bash
set -euo pipefail

# EXP004 — Val-only entry point; run after human approval.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-${REPO_ROOT}/../../data/ActiveView}"
EXP_NAME="EXP004_radius_ablation"
OUT="${DATA_ROOT}/experiments/stage_c_v1/${EXP_NAME}"
FEATURE_ROOT="${DATA_ROOT}/datasets/policy_v11_5/stage_c_v1/${EXP_NAME}"

mkdir -p "${OUT}/checkpoints" "${OUT}/runtime"
git -C "${REPO_ROOT}" rev-parse HEAD > "${OUT}/code_commit.txt"
cp "${SCRIPT_DIR}/config.yaml" "${OUT}/config.yaml"

(cd "${REPO_ROOT}" && python -m activeview.scripts.build_stage_c_variant_cache \
  --variant radius_ablation \
  --source-feature-root "${DATA_ROOT}/datasets/policy_v11_5/stage_c" \
  --output-dir "${FEATURE_ROOT}")

(cd "${REPO_ROOT}" && python -m activeview.scripts.train_stage_c \
  --model-type set_ranker \
  --feature-root "${FEATURE_ROOT}" \
  --stage-b-root "${DATA_ROOT}/datasets/policy_v11_5/stage_b" \
  --output-dir "${OUT}/checkpoints" \
  --sampler record_balanced \
  --episodes-per-record 16 \
  --batch-size 128 \
  --max-epochs 100 \
  --patience 10 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --lambda-reg 1.0 \
  --lambda-rank 1.0 \
  --tau 0.5 \
  --lambda-gap 0.0 \
  --seed 42)

(cd "${REPO_ROOT}" && python -m activeview.scripts.evaluate_stage_c_val \
  --model-type set_ranker \
  --feature-root "${FEATURE_ROOT}" \
  --stage-b-root "${DATA_ROOT}/datasets/policy_v11_5/stage_b" \
  --dataset-root "${DATA_ROOT}/datasets/policy_v11_5" \
  --checkpoint "${OUT}/checkpoints/set_ranker_best.pth" \
  --output-dir "${OUT}/runtime" \
  --baseline "${SCRIPT_DIR}/baseline.json" \
  --experiment-id EXP004 > "${OUT}/result.json")
