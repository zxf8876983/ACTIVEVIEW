# ACTIVEVIEW Handoff

Status: DOCUMENTED / READY FOR NEXT COMMAND
Updated: 2026-08-27

## Canonical v11.5

The active implementation is `ea_avs_mvp_v11/`. The protocol is selected16: audited official-150 14 classes plus `lie` and `stumble`; `fall` is excluded. BABEL `train.json` and `val.json` are used directly after single-label filtering, strict `num_frames > 30`, conflict removal, official Train/Val caps 400/100, and seed 42.

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
- Frozen ST-GCN: `/home/zxf/WorkSpace/code/data/ActiveView/checkpoints/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled/`.
- YOLO: `/home/zxf/WorkSpace/code/data/ActiveView/checkpoints/ultralytics/yolo26n-pose.pt`.
- Habitat/semantic root: `/home/zxf/WorkSpace/code/code/robot/DATA/` only.
- Humanoid: `/home/zxf/WorkSpace/code/data/ActiveView/assets/habitat_humanoids/male_0/`.

## Offline active-view data

Data is under `datasets/offline/<scene-set>/<original-scene-folder>/`. Four furniture-based regions are used: `bedroom`, `living_room`, `kitchen`, `dining_area`. Each placement has 32 views (1.5/2.0/2.5/3.0 m × 8 azimuths). `semantic-region-v2` candidate manifests and `semantic-region-offline-v2` records persist all geometry, camera, static navigation and 32 skeleton/confidence fields; RGB/Depth are not saved.

Static placement reachability is not sufficient for a trajectory. Evaluation recomputes `ShortestPath(P_current, P_candidate)` from the robot's actual current position before choosing a next view. Current evaluators report `NoMove`, `Fixed`, `Random`, `Nearest` and hindsight candidate-pool `Oracle`; v11.3 Utility Predictor code/checkpoints are not active.

As of this handoff, the 21-scene HM3D-train orchestration has 12 complete scene manifests and one incomplete scene directory. No generation/evaluation process is running. Resume only with the scene-level orchestrator and v2 schema checks.

## Historical boundary

`ea_avs_mvp_v10/` is read-only reference. It contains candidate generation and per-view entropy analysis, not a learned active-view selector. The old v11.3 `active_view`/Utility Predictor modules are deleted from the working tree but remain recoverable from Git history until deletion is committed.

## Validation and next step

Documentation and state files were updated to match the current scripts and runtime metadata. No experiment was run in this documentation task. Before future code changes, run a lightweight `compileall`/focused test set, then update this handoff with the new commit and process state.
