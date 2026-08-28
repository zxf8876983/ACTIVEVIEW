# ACTIVEVIEW Current Task

## Status

**DOCUMENTED / READY FOR NEXT COMMAND** — v11.5 canonical selected16 data, frozen ST-GCN, semantic-region offline schema v2 and dynamic-reachability evaluators are documented below. The corrected HM3D-train scene generator is currently running under `conda habitat` with four workers; no evaluation process is running.

## Current truth

- Active source: `ea_avs_mvp_v11/`.
- Train/Val data: `/home/zxf/WorkSpace/code/data/ActiveView/datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/` (3,240/980, `(N,3,30,17,1)`).
- Frozen checkpoint: `/home/zxf/WorkSpace/code/data/ActiveView/checkpoints/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled/`.
- Active pose chain: RGB-only 256×256 → Ultralytics YOLO26n-Pose → VideoPose3D → camera-to-gravity/YZ conversion → root/scale/yaw-only → ST-GCN.
- Habitat/semantic root: `/home/zxf/WorkSpace/code/code/robot/DATA/` only.
- Humanoid: `/home/zxf/WorkSpace/code/data/ActiveView/assets/habitat_humanoids/male_0/`.
- Offline data root: `/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/` with `hm3d-minival/` and `hm3d-train/` scene-set folders.

## ST-GCN protocol

The fixed 16-class mapping is official-150 audited 14 classes plus `lie` and `stumble`; `fall` is not active. BABEL `train.json`/`val.json` are split directly, with single-label filtering, strict `num_frames > 30`, conflicting source-interval removal, official caps 400/100, auxiliary classes uncapped, and seed 42. ST-GCN receives only estimated H36M-17 skeletons. Training uses tempered oversampling and class-weighted cross entropy; `ReduceLROnPlateau` and early stopping monitor deterministic full-Train loss only. The frozen checkpoint is the final stopped-epoch model; Val is evaluated once post-training for upper-bound diagnosis and is never used for checkpoint selection or policy training.

Stage A policy records use canonical `train/val/test = 6:2:2`; the persisted
split `summary.json -> split_ratios` is the single source consumed by the
Episode builder.

## Offline strategy protocol

Each semantic scene/region has one furniture-based human placement and 32 candidate viewpoints (radii 1.5/2.0/2.5/3.0 m × eight azimuths). Offline generation uses four COLOR cameras per worker, RGB-to-skeleton inference, and stores skeleton/confidence plus scene ID, navmesh path, placement, raw/snapped/actual agent positions, rotations, navigability, placement-referenced reachability and costs. No RGB/Depth is saved. The schema is `semantic-region-offline-v2`; candidate metadata is `semantic-region-v2`.

The placement reachability flag is static metadata only. During sequential evaluation, Habitat reloads the navmesh and recomputes paths from the robot's current position to every pending candidate before selecting. Current policies are `NoMove`, `Fixed`, `Random`, `Nearest`, and hindsight candidate-pool `Oracle`; no learned Utility Predictor is implemented or evaluated.

## Current data status

- HM3D-minival: `offline/hm3d-minival/00800-TEEsavR23oF/` is the canonical minival scene.
- HM3D-train: 21-scene selection is recorded in `offline/hm3d-train/dataset_summary.json`; all 21 scene folders have complete 980×4×32 manifests. `00592-CthA7sQNTPK` and `00643-ggNAcMh8JPT` have been rotation-audited against the exact offline render state. `00643` required metadata refresh only (`npz_changed=0`); no generation process remains active.
- Stage A policy split has been regenerated with canonical `train/val/test = 6:2:2`: 589/197/194 records, 980 unique records total. Existing serialized Stage A Episodes still reflect the former split and must be rebuilt before policy evaluation.
- Dynamic, random-start and grid-start evaluation outputs remain under `results/` with their corresponding caches under `datasets/strategy_eval_cache/`.

## Canonical entry points

```text
scripts/prepare_selected16_manifests.py
scripts/generate_selected16_habitat_dataset.py
scripts/generate_selected16_habitat_parallel.py
scripts/train_selected16_habitat_stgcn.py
scripts/generate_semantic_region_candidate_metadata.py
scripts/generate_semantic_region_offline_views.py
scripts/generate_hm3d_train_four_region_offline.py
scripts/evaluate_semantic_region_offline.py
scripts/evaluate_hm3d_train_dynamic_reachability.py
scripts/evaluate_hm3d_train_random_initializations.py
scripts/evaluate_hm3d_train_grid_initializations.py
```

## Invariants

1. Never use historical datasets/checkpoints as current defaults.
2. Never feed AMASS/SMPL GT joints into ST-GCN.
3. Keep RGB-only, YOLO26n-Pose, VideoPose3D, 30 frames, H36M-17 and yaw-only alignment.
4. Keep `fall` excluded and `lie`/`stumble` included unless the user changes the protocol.
5. Do not scan `/home/zxf/MG08/` or any undeclared scene root.
6. Do not let future candidate RGB, labels or post-hoc ST-GCN predictions enter an executable policy decision.

## Stage A acceptance

The final Episode JSONL is audited after serialization with
`ea_avs_mvp_v11/scripts/validate_stage_a.py`. The audit checks split isolation,
current/candidate validity, record-local cached skeleton paths, finite geometry
and costs, episode uniqueness, NPZ/Episode geometry correspondence, and
recursive future-perception leakage fields. Individual non-finite skeleton
viewpoints are allowed when they are not selected; only the final
current/candidate IDs must be finite. The default command also validates every
referenced cached NPZ archive. Real Habitat NavMesh
ShortestPath verification is an explicit second step:

```bash
conda run --no-capture-output -n habitat python ea_avs_mvp_v11/scripts/validate_stage_a.py --verify-habitat
```

Unit tests do not replace this Habitat integration check. If the Habitat
dependency or scene assets are unavailable, the result must be reported as
`NOT RUN`, not as a passed Stage A acceptance.
