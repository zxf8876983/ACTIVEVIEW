# Reduced12 clean-perception reference recovery

Val-only, read-only audit. No policy Test, training, RGB rendering, YOLO, VideoPose3D, or skeleton regeneration was performed.

## Findings

1. Historical tensor found: `raw-train/val_data.npy` with its paired `val_labels.npy` and `val_metadata.json`. It is the tensor consumed by the recorded reduced12 ST-GCN validation protocol.
2. Frozen-checkpoint reproduction: `0.767346939` Accuracy / `0.781068690` Macro-F1; recorded values are `0.767346939` / `0.781068690` (absolute differences `0` / `0`).
3. Exact mapping to the current 105 Official-Val Moving records (the Stage-D Val file contains 10080 contexts): 0/105. The historical clean tensor is Official-Train-derived, while current ActiveView Moving-Val records are Official-Val-derived; no record id or full source interval matched.
4. Because exact 105-record mapping is unavailable, the candidate-level clean-reference similarity and privileged selector audit is stopped. The earlier Habitat-FK reference must not be interpreted as a substitute.

## Provenance

- Frozen checkpoint: `/home/zxf/WorkSpace/code/data/ActiveView/checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/stgcn_reduced12_no_kneel_clean_best.pth`
- Historical dataset root: `/home/zxf/WorkSpace/code/data/ActiveView/datasets/reduced12_no_kneel_clean_babel_diversity_v1/raw-train`
- Label order exact: `True`
- Shape/metadata checks: `True` / `True` / finite=`True`
- Preprocessing evidence complete: `True`
- Pipeline evidence: RGB → YOLO26n-Pose → VideoPose3D → camera_to_gravity → root_center → torso_scale → yaw_only → SkeletonNormalizer(align_canonical=True).

## Missing asset for the requested Moving-Val audit

The repository has no clean-perception tensor/metadata for `raw-val/val.json` (the 105 records used by ActiveView Val). Expected files are `raw-val/val_data.npy`, `raw-val/val_labels.npy`, and `raw-val/val_metadata.json` under the reduced12 dataset root. They must be recovered from the historical archive before any 105-record or 10,080-context clean comparison can be run.

## Flags

```yaml
test_used: false
training_used: false
new_rgb_rendered: false
new_pose_estimation: false
historical_clean_perception_only: true
exact_record_mapping_required: true
clean_reference_valid_only_if_reproduction_passes: true
deployable: false
```
