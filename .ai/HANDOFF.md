# ACTIVEVIEW Handoff

Status: DOCUMENTED / READY FOR NEXT COMMAND
Updated: 2026-08-28

## Canonical v11.5

The active implementation is `ea_avs_mvp_v11/`. The protocol is selected16: audited official-150 14 classes plus `lie` and `stumble`; `fall` is excluded. BABEL `train.json` and `val.json` are used directly after single-label filtering, strict `num_frames > 30`, conflict removal, official Train/Val caps 400/100, and seed 42. Stage A policy records use `train/val/test = 6:2:2`, read from persisted split `summary.json`.

The perception chain is:

```text
AMASS/SMPL → male_0 pure-color Habitat → RGB-only 256×256
→ Ultralytics YOLO26n-Pose → VideoPose3D
→ Human3.6M Y/Z conversion + Habitat camera-to-gravity
→ root center + torso scale + yaw-only → H36M-17 ST-GCN
```

ST-GCN never receives AMASS/SMPL GT joints. Grounding uses URDF visual geometry and supporting-floor raycast; grounding offsets are cached per action. Yaw-only alignment preserves gravity-related roll/pitch.

## Canonical data and model

- Train/Val: `/home/zxf/WorkSpace/code/data/ActiveView/datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/` (3,240/980, `(N,3,30,17,1)`).
- Frozen ST-GCN: `/home/zxf/WorkSpace/code/data/ActiveView/checkpoints/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled/`. The canonical checkpoint is the final stopped-epoch model from Train-only convergence; the former Val-Macro-F1 legacy weight was removed.
- YOLO: `/home/zxf/WorkSpace/code/data/ActiveView/checkpoints/ultralytics/yolo26n-pose.pt`.
- Habitat/semantic root: `/home/zxf/WorkSpace/code/code/robot/DATA/` only.
- Humanoid: `/home/zxf/WorkSpace/code/data/ActiveView/assets/habitat_humanoids/male_0/`.

## Offline active-view data

Data is under `datasets/offline/<scene-set>/<original-scene-folder>/`. Four furniture-based regions are used: `bedroom`, `living_room`, `kitchen`, `dining_area`. Each placement has 32 views (1.5/2.0/2.5/3.0 m × 8 azimuths). `semantic-region-v2` candidate manifests and `semantic-region-offline-v2` records persist all geometry, camera, static navigation and 32 skeleton/confidence fields; RGB/Depth are not saved.

Static placement reachability is not sufficient for a trajectory. Evaluation recomputes `ShortestPath(P_current, P_candidate)` from the robot's actual current position before choosing a next view. Current evaluators report `NoMove`, `Fixed`, `Random`, `Nearest` and hindsight candidate-pool `Oracle`; v11.3 Utility Predictor code/checkpoints are not active.

All 21 requested HM3D-train scene manifests are complete. Each scene contains
980 records × 4 regions × 32 viewpoints. `00592-CthA7sQNTPK` and
`00643-ggNAcMh8JPT` were rotation-audited and matched the exact offline render
state; `00643` required only metadata refresh (`npz_changed=0`). No scene
generation process remains active.

The Stage A policy split was regenerated with canonical 6:2:2 ratio
(589/197/194 records; 980 unique records). Existing serialized Episode JSONL
from the previous 70/15/15 split was intentionally not regenerated in this
step; rebuild Stage A Episodes before evaluation.

## Historical boundary

`ea_avs_mvp_v10/` is read-only reference. It contains candidate generation and per-view entropy analysis, not a learned active-view selector. The old v11.3 `active_view`/Utility Predictor modules are deleted from the working tree but remain recoverable from Git history until deletion is committed.

## Validation and next step

The ST-GCN trainer now uses full-Train loss for scheduling and early stopping, saves the final stopped-epoch weights, and evaluates Val once post-training. The old Val-Macro-F1 checkpoint was removed; v10 historical ST-GCN modules remain read-only reference code. Before future code changes, run a lightweight `compileall`/focused test set, then update this handoff with the new commit and process state.

## Stage A audit hardening (2026-08-28)

`policy_episode_builder.audit_episode_files()` now derives integrity flags
from serialized Episode JSONL rather than hard-coded values. It checks split
isolation, current/candidate IDs and geometry, record-local skeleton paths,
candidate costs, and recursive future-perception leakage fields. With
`validate_cached_skeletons=True`, it validates every referenced NPZ shape,
navigation arrays, viewpoint IDs, and finite skeleton frames. The read-only
acceptance entry point is `ea_avs_mvp_v11/scripts/validate_stage_a.py`;
`--verify-habitat` recomputes real HM3D `ShortestPath` for final Episodes.
The lightweight JSONL audit passes for 59,780 Episodes (no duplicate episode
keys/IDs and no integrity failures). The cached-NPZ audit now additionally
cross-checks Episode geometry against the archive; a new full run after this
change was not completed because it is I/O-heavy. One real Habitat Episode
smoke check previously passed. A full all-Episode Habitat path replay remains
an explicit, potentially expensive acceptance run and has not been executed.

The Stage A audit now allows partial non-finite skeleton viewpoints. It records
`nonfinite_cached_skeleton_viewpoints` for diagnosis, while current/candidate
validity is checked against the per-archive finite ID set. It also checks
`record_id × scene_id × region` and `episode_id` uniqueness, NPZ/Episode
geometry correspondence, expanded future-perception field leakage, and
label/label_id consistency in policy splits.

The latest audit hardening also validates split summary metadata against the
actual `train.json`/`val.json`/`test.json` files (counts, unique IDs,
per-class counts, and canonical 6:2:2 ratios). Coverage now uses the target
scene list before generation, separately audits scene-level failures, and
requires `all_target_scenes_used=true`; a failed scene cannot be silently
removed from the expected tuple set.
