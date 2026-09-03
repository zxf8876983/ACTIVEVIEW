# RGB Observation Dataset V1

RGB Observation V1 restores one raw Habitat color frame for every canonical
HM3D-train skeleton record and viewpoint. The canonical skeleton tree is
read-only; RGB files are written to a separate MG08 data root and are not
committed to Git.

## Protocol

- Fixed zero-based motion frame: `15` of the canonical 30-frame sequence.
- One compressed NPZ per record, with `rgb` shaped `[32, 256, 256, 3]` and
  dtype `uint8`.
- Viewpoint IDs, camera positions and rotations are copied from the source
  skeleton NPZ without reordering or recomputation.
- Canonical motion conversion, grounding and frame-15 humanoid restoration are
  reused. No perception or skeleton inference is run.
- Generation used 16 independent Habitat workers per scene, as explicitly
  authorized during execution. Scenes were processed sequentially.

## Frozen paths and provenance

| Item | Location |
|---|---|
| Source skeleton root | `/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/` |
| RGB output root | `/home/zxf/MG08/robot/ActiveView/datasets/offline/hm3d-train/` |
| Generator | `activeview/scripts/data/generate_hm3d_train_rgb_observations.py` |
| Motion manifest SHA256 | `c20fabd3ce11c31b2af4712b2e5e59fa743d4765419bf3b9089a0cde92227b61` |

The external output root contains `dataset_summary.json` and one
`rgb_manifest.json` per scene. The Git-tracked compact snapshot is
`rgb_observation_v1_summary.json` beside this report.

## Completeness and storage audit

- Scenes: **21**
- Source skeleton records: **82,320**
- RGB records: **82,320**
- RGB viewpoints: **2,634,240**
- Missing / extra / invalid records: **0 / 0 / 0**
- Output bytes (measured): **214,924,343,585** (200.16 GiB)
- Mean compressed NPZ size: **2,610,486.88 bytes/record**
- Mean compressed RGB frame size: **81,577.72 bytes/frame**
- Temporary files after completion: **0**

An initial smoke run covered one scene and two records per region; all 256
views passed schema, camera metadata, source hash and nonzero-pixel checks.
The final full audit repeated these checks for every record.

## Non-destructive guarantees

`skeleton_modified=false`, `skeleton_regenerated=false`, `yolo_used=false`,
`videopose3d_used=false`, and `stgcn_used=false`. No policy Test evaluation,
RGB embedding extraction or downstream experiment was started.
