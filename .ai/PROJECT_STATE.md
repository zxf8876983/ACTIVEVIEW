# ACTIVEVIEW Project State

Last Updated: 2026-08-28
Active Version: v11.5 (`ea_avs_mvp_v11/`)

## Canonical scientific protocol

ACTIVEVIEW 研究机器人在 HM3D 室内环境中选择下一个观察视点，以降低人体姿态/动作识别不确定性。ST-GCN 是冻结的下游 backbone，不是本项目的创新对象；v11.5 Stage C 已实现 current-conditioned learned Utility Predictor，但尚未进入 Stage D 或 Habitat 在线 learned-policy 评估。

- Labels: 16 classes = audited official-150 subset (`t pose`, `cartwheel`, `knock`, `play instrument`, `crawl`, `a pose`, `kick`, `sit`, `move up/down incline`, `jog`, `stand up`, `jump`, `walk`, `throw`) + `lie` + `stumble`; `fall` is not an active label.
- Split: BABEL `train.json` → Train and `val.json` → Val. Only single-label intervals with `num_frames > 30` survive; conflicting identical source intervals are removed. Official classes are capped at Train 400 / Val 100; `lie` and `stumble` are uncapped.
- Policy split: canonical `train/val/test = 6:2:2`; the persisted split `summary.json -> split_ratios` is the single source read by the Stage A Episode builder.
- Perception: AMASS/SMPL → project-local `male_0` in pure-color Habitat → RGB-only 256×256 → Ultralytics YOLO26n-Pose → VideoPose3D → Y/Z camera conversion and camera-to-gravity → root center + torso scale + yaw-only normalization → H36M-17 ST-GCN.
- ST-GCN never receives AMASS/SMPL GT joints. Yaw-only alignment preserves gravity-relative roll/pitch, so lying/fall-like posture is not rotated upright.

## Canonical runtime artifacts

Runtime root: `/home/zxf/WorkSpace/code/data/ActiveView/` (override with `ACTIVEVIEW_DATA_ROOT`).
Habitat/semantic root: `/home/zxf/WorkSpace/code/code/robot/DATA/` (override with `ACTIVEVIEW_HABITAT_DATA_ROOT`; never scan other disks).
Humanoid: `/home/zxf/WorkSpace/code/data/ActiveView/assets/habitat_humanoids/male_0/`.

- Train/Val estimated skeletons: `datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/` (3,240/980; `(N,3,30,17,1)`).
- Frozen checkpoint: `checkpoints/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled/`.
- YOLO weights: `checkpoints/ultralytics/yolo26n-pose.pt`.
- Offline strategy data: `datasets/offline/hm3d-minival/00800-TEEsavR23oF/` and `datasets/offline/hm3d-train/<original-scene-folder>/`; RGB/Depth are not saved. All 21 selected HM3D-train scenes are complete.
- Offline schema: candidate manifest `semantic-region-v2`; record schema `semantic-region-offline-v2`.

## Dataset generation and training entry points

- Manifest: `scripts/prepare_selected16_manifests.py`.
- RGB/Habitat generation: `scripts/generate_selected16_habitat_dataset.py` and `generate_selected16_habitat_parallel.py`.
- Core generation: `dataset/babel_selected16_manifest.py`, `dataset/babel_clean_dataset_generator.py`, `dataset/humanoid_grounding.py`.
- Pose and normalization: `perception/ultralytics_pose3d_estimator.py`, `perception/skeleton_normalizer.py`.
- ST-GCN training: `scripts/train_selected16_habitat_stgcn.py`.

Generation defaults are RGB-only, COLOR sensor, 256×256, uniform 30-frame sampling, YOLO26n-Pose and VideoPose3D. Grounding uses URDF visual geometry and supporting-floor raycast; offsets are precomputed once per action and reused for all views. Training uses tempered `count^-0.5` `WeightedRandomSampler`, normalized `sqrt(N/count)` class weights, Adam, and `ReduceLROnPlateau` on deterministic full-Train loss. Early stopping is Train-loss-only (`max_epochs=200`, `patience=20`, `min_delta=1e-4`); Val is read only once after training for post-hoc upper-bound diagnosis and never selects a checkpoint.

## Offline active-view data and evaluation

Each scene has four furniture-based semantic placements: `bedroom`, `living_room`, `kitchen`, `dining_area`. Each placement has 4 radii (1.5/2.0/2.5/3.0 m) × 8 azimuths = 32 candidate viewpoints. Each record stores scene/navmesh/placement identifiers, raw/snapped/actual agent positions, camera rotations, static navigability/reachability, path costs, 32 estimated skeletons and confidence values.

The static `is_reachable_from_placement` flag is only a reference. Sequential evaluation reloads the navmesh and recomputes `ShortestPath(P_current, P_candidate)` before selecting the next view. Current policies are `NoMove`, `Fixed`, `Random`, `Nearest`, and hindsight candidate-pool `Oracle`; Oracle can inspect the true label only after the dynamic pool is formed and is not executable. Stage C checkpoints are under `checkpoints/stage_c/`; they use only current frozen-ST-GCN features plus candidate geometry, with no future perception input, body yaw or movement penalty.

Entrypoints:

- Candidate metadata: `scripts/generate_semantic_region_candidate_metadata.py`.
- Offline skeleton views: `scripts/generate_semantic_region_offline_views.py`.
- 21-scene HM3D-train orchestration: `scripts/generate_hm3d_train_four_region_offline.py`.
- Static baseline: `scripts/evaluate_semantic_region_offline.py`.
- Dynamic reachability: `scripts/evaluate_hm3d_train_dynamic_reachability.py`.
- Cached 500 continuous starts: `scripts/evaluate_hm3d_train_random_initializations.py`.
- Cached 32-grid starts: `scripts/evaluate_hm3d_train_grid_initializations.py`.

As of this update, all 21 HM3D-train scene folders have complete manifests
(980 actions × 4 regions × 32 views), and no generation process is running.
The scene list and status are recorded in
`datasets/offline/hm3d-train/dataset_summary.json`.

## Evaluation records

- Static minival baseline: `results/semantic_region_offline_baselines.json`.
- Dynamic ten-scene result: `results/hm3d_train_dynamic_reachability_10scenes.json` (NoMove 39,200 records; movement policies 38,220; Fixed 39.00%, Random 34.84%, Nearest 40.33%, Oracle 88.91%).
- Current Train-only 500-start cache/result: `datasets/strategy_eval_cache/hm3d-train_random_init_500_train_only_v2/`, `results/hm3d_train_random_initializations_500_train_only.json` (17,150,000 records; NoMove 39.49%, Fixed 40.10%, Random 34.95%, Nearest 39.61%, Oracle 92.44%; 187.1 s).
- Current Train-only 32-grid cache/result: `datasets/strategy_eval_cache/hm3d-train_grid_init_32_train_only_v2/`, `results/hm3d_train_grid_initializations_32_train_only.json` (253,820 records; NoMove 39.76%, Fixed 40.54%, Random 34.63%, Nearest 41.68%, Oracle 93.95%; 130.8 s).
- The unsuffixed random/grid result files are retained as historical pre-Train-only comparisons and are not canonical.

## Stage C current-conditioned utility prediction

Stage C is implemented from frozen Stage A/B artifacts. Feature cache and
evaluation outputs are under `datasets/policy_v11_5/stage_c/`; current input
is a 275-D frozen ST-GCN state and each candidate input is 11-D geometry.
`PairwiseUtilityMLP` (142,785 parameters, selected epoch 17) and
permutation-equivariant `SetUtilityRanker` (407,745 parameters, selected epoch
28) were trained with record-balanced sampling and Val Macro-F1 checkpoint
selection. Their combined summary is `stage_c/stage_c_summary.json`, and the
independent validator report is `stage_c/validation_report.json` with
`passed=true`. Test metrics are final-only diagnostics; Stage D and Habitat
online learned-policy evaluation have not started.

## Historical code boundary

The v11 working tree intentionally removes old v10 RGB-D/legacy pose entry points, old household/official150 training entry points, and v11.3 Utility Predictor/closed-loop modules. `ea_avs_mvp_v10/` remains read-only historical reference; it has candidate generation and per-view entropy analysis, but no learned active-view selector.
