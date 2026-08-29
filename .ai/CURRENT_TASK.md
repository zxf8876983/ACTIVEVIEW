# Current Task

## Current task

EXP004–EXP007 Train→Val runs are complete and ready for joint human review.
All four were independent Val-only experiments against the frozen Stage C-v0
baseline. Compact results are recorded in each experiment directory and full
runtime outputs remain under `ACTIVEVIEW_DATA_ROOT`.

## Frozen

Stage A/B/C-v0 artifacts, ST-GCN, split, candidate pool, current viewpoint,
decision protocol, and all accepted runtime data. EXP003 is rejected; its
diagnostics are evidence only and are not a baseline or a dependency.

## Do not

- run Test;
- regenerate Habitat, YOLO, VideoPose3D, Stage A/B/C-v0 or EXP003 artifacts;
- create EXP008 or enter Stage D.
