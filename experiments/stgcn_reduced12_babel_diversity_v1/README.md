# Reduced-12 BABEL diversity-aware ST-GCN

This is an independent 12-class dataset and recognizer. The frozen
selected-16 dataset is not modified.

## Classes

`walk`, `sit`, `stand up`, `bend`, `squat`, `lean`, `stretch`,
`take/pick something up`, `place something`, `lift something`, `clean something`,
`stumble`.

## Protocol

- BABEL official `train.json`: at most 300 records per class, selected with a
  deterministic diversity-first policy (unique source, subject, AMASS dataset,
  and duration bin), then split 90%/10% into ST-GCN development train/val.
- BABEL official `val.json`: at most 100 records per class with the same policy,
  then split 60%/20%/20% into ActiveView motion train/val/test.
- Seed: 42. Duration bins are `<1`, `1-2`, `2-4`, `4-8`, and `>=8` seconds.
- Generated tensors use the existing frozen RGB → YOLO26n-Pose → VideoPose3D →
  gravity/root/scale/yaw preprocessing; no selected-16 files are overwritten.

## Runtime artifacts

Generated data is outside Git at
`/home/zxf/WorkSpace/code/data/ActiveView/datasets/reduced12_babel_diversity_v1/`.
The ST-GCN checkpoint is at
`/home/zxf/WorkSpace/code/data/ActiveView/checkpoints/stgcn_reduced12_babel_diversity_v1/stgcn_reduced12_best.pth`.

Generation entry point:
`activeview/scripts/data/prepare_reduced12_dataset.py` followed by
`activeview/scripts/data/generate_reduced12_skeleton_dataset.py`.
Training entry point:
`activeview/scripts/train/train_selected16_habitat_stgcn.py` with the reduced12
data root.

See `config.yaml`, `result.json`, and `analysis.md` for the frozen record
counts, diversity statistics, hashes, and training outcome.
