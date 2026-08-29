#!/usr/bin/env bash
set -euo pipefail

# EXP009 — Train-to-Val only; execute after human approval.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-${REPO_ROOT}/../../data/ActiveView}"
FEATURE_ROOT="${DATA_ROOT}/datasets/policy_v11_5/stage_c_v2/joint_tokens"
OUT="${DATA_ROOT}/experiments/stage_c_v2/EXP009_candidate_conditioned_attention"
BASELINE="${SCRIPT_DIR}/baseline.json"
CHECKPOINT="${OUT}/checkpoints/candidate_conditioned_attention_best.pth"

mkdir -p "${OUT}/checkpoints" "${OUT}/runtime"
git -C "${REPO_ROOT}" rev-parse HEAD > "${OUT}/code_commit.txt"
cp "${SCRIPT_DIR}/config.yaml" "${OUT}/config.yaml"
if [[ ! -f "${FEATURE_ROOT}/stage_c_v2_feature_summary.json" ]]; then
  (cd "${REPO_ROOT}" && python -m activeview.scripts.build_stage_c_v2_cache --output-dir "${FEATURE_ROOT}" --splits train val)
fi
(cd "${REPO_ROOT}" && python -m activeview.scripts.train_stage_c_v2 \
  --model-type candidate_conditioned_attention --feature-root "${FEATURE_ROOT}" \
  --stage-b-root "${DATA_ROOT}/datasets/policy_v11_5/stage_b" \
  --output-dir "${OUT}/checkpoints" --batch-size 128 --episodes-per-record 16 \
  --max-epochs 100 --patience 10 --lr 0.001 --weight-decay 0.0001 \
  --lambda-reg 1.0 --lambda-rank 1.0 --tau 0.5 --seed 42)
(cd "${REPO_ROOT}" && python -m activeview.scripts.evaluate_stage_c_v2_val \
  --model-type candidate_conditioned_attention --feature-root "${FEATURE_ROOT}" \
  --source-feature-root "${DATA_ROOT}/datasets/policy_v11_5/stage_c" \
  --stage-b-root "${DATA_ROOT}/datasets/policy_v11_5/stage_b" \
  --dataset-root "${DATA_ROOT}/datasets/policy_v11_5" --checkpoint "${CHECKPOINT}" \
  --baseline "${BASELINE}" --output-dir "${OUT}/runtime" --experiment-id EXP009 \
  > "${OUT}/result.json")
