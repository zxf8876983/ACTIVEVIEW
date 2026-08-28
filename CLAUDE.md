# ACTIVEVIEW Repository Guidance

`activeview/` is the sole production and research source package. Directory-
versioned development (`ea_avs_mvp_v*`, `activeview_v*`) is retired; historical
versions are recoverable from Git history and are not part of the working tree.

## Runtime boundaries

- Source root: this repository.
- Runtime data: `ACTIVEVIEW_DATA_ROOT` or the default sibling
  `../../data/ActiveView/`.
- Habitat assets: `ACTIVEVIEW_HABITAT_DATA_ROOT` or
  `/home/zxf/WorkSpace/code/code/robot/DATA/`; never scan other disks.
- Large datasets, checkpoints, RGB/depth, skeletons, caches, and predictions
  stay outside Git and are covered by `.gitignore`.

## Canonical v11.5 pipeline

The active pipeline is selected16 BABEL data → project-local `male_0` →
RGB-only 256×256 → YOLO26n-Pose → VideoPose3D → camera/gravity conversion →
root/scale/yaw-only normalization → H36M-17 ST-GCN. ST-GCN receives estimated
skeletons only. Stage A/B/C artifacts are frozen and must not be regenerated
or altered during repository-only maintenance.

## Research constraints

- Candidate selection cannot use future RGB/depth, labels, or post-hoc
  predictions.
- Use `activeview.core.paths` for runtime paths; do not hardcode new machine
  paths.
- Preserve accepted protocol, split, candidate pool, and provenance.
- Do not train, regenerate data, or run Test tuning as part of structural
  cleanup.

## Commands

Package-bound entry points live under `activeview/scripts/`, for example:

```bash
python -m activeview.scripts.validate_stage_a
python -m activeview.scripts.validate_stage_b
python -m activeview.scripts.validate_stage_c
pytest -q tests/unit tests/integration
```

The repository-level `tests/` directory contains unit and integration tests.
Historical root Markdown reports may retain old version names as factual
history, but active commands and documentation must use `activeview/...`.
