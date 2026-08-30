#!/usr/bin/env bash
set -euo pipefail

# EXP014 is intentionally Train -> Val only. Review and authorize before use.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-${REPO_ROOT}/../../data/ActiveView}"
DATASET_ROOT="${DATA_ROOT}/datasets/policy_v11_5"
STAGE_B_ROOT="${DATASET_ROOT}/stage_b"
FEATURE_ROOT="${DATASET_ROOT}/stage_c"
STAGE_D_CACHE="${DATA_ROOT}/datasets/policy_v11_5/stage_d/EXP014_two_step_sequential"
V0_PREDICTIONS="${DATA_ROOT}/experiments/stage_d/EXP014_two_step_sequential/v0_predictions"
OUT="${DATA_ROOT}/experiments/stage_d/EXP014_two_step_sequential"
PAIRWISE_ROOT="${DATA_ROOT}/datasets/policy_v11_5/pairwise_viewpoint_geodesic"
STGCN_CHECKPOINT="${DATA_ROOT}/checkpoints/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled_best.pth"
LABEL_MAPPING="${DATA_ROOT}/datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/label_mapping.json"

mkdir -p "${OUT}/checkpoints" "${OUT}/runtime"
git -C "${REPO_ROOT}" rev-parse HEAD > "${OUT}/code_commit.txt"
cp "${SCRIPT_DIR}/config.yaml" "${OUT}/config.yaml"

(cd "${REPO_ROOT}" && python -m activeview.scripts.generate_stage_c_v0_predictions \
  --feature-root "${FEATURE_ROOT}" --stage-b-root "${STAGE_B_ROOT}" \
  --checkpoint "${DATA_ROOT}/checkpoints/stage_c/set_ranker_best.pth" \
  --output-dir "${V0_PREDICTIONS}" --device cuda:0 --batch-size 256)
HABITAT_ROOT="${ACTIVEVIEW_HABITAT_DATA_ROOT:?Set ACTIVEVIEW_HABITAT_DATA_ROOT to the declared Habitat DATA root}"
SCENE_ROOT="${HABITAT_ROOT}/hm3d-train"
for SCENE_ID in $(python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(" ".join(p["scene_ids_used"]))' "${DATASET_ROOT}/stage_a_summary.json"); do
  if [ ! -f "${PAIRWISE_ROOT}/${SCENE_ID}/manifest.json" ]; then
    (cd "${REPO_ROOT}" && python -m activeview.scripts.build_pairwise_viewpoint_geodesic \
      --scene-dir "${SCENE_ROOT}/${SCENE_ID}" \
      --candidate-manifest "${DATA_ROOT}/datasets/offline/hm3d-train/${SCENE_ID}/candidate_metadata/manifest.json" \
      --output-root "${PAIRWISE_ROOT}/${SCENE_ID}")
  fi
done
(cd "${REPO_ROOT}" && python -m activeview.scripts.build_stage_d_cache \
  --dataset-root "${DATASET_ROOT}" --stage-b-root "${STAGE_B_ROOT}" \
  --feature-root "${FEATURE_ROOT}" \
  --train-predictions "${V0_PREDICTIONS}/train_predictions.jsonl" \
  --val-predictions "${V0_PREDICTIONS}/val_predictions.jsonl" \
  --pairwise-root "${PAIRWISE_ROOT}" --checkpoint "${STGCN_CHECKPOINT}" \
  --output-dir "${STAGE_D_CACHE}" --device cuda:0)
(cd "${REPO_ROOT}" && python -m activeview.scripts.train_stage_d \
  --cache-root "${STAGE_D_CACHE}" --stage-b-root "${STAGE_B_ROOT}" \
  --checkpoint-source "${STGCN_CHECKPOINT}" --label-mapping "${LABEL_MAPPING}" \
  --output-dir "${OUT}/checkpoints" --device cuda:0 --batch-size 128 \
  --episodes-per-record 16 --max-epochs 100 --patience 10 --lr 0.001 \
  --weight-decay 0.0001 --seed 42)
(cd "${REPO_ROOT}" && python -m activeview.scripts.evaluate_stage_d_val \
  --cache-root "${STAGE_D_CACHE}" --stage-b-root "${STAGE_B_ROOT}" \
  --checkpoint "${OUT}/checkpoints/sequential_observation_ranker_best.pth" \
  --v0-predictions "${V0_PREDICTIONS}/val_predictions.jsonl" \
  --label-mapping "${LABEL_MAPPING}" --output-dir "${OUT}/runtime" \
  --device cuda:0 --batch-size 128)
