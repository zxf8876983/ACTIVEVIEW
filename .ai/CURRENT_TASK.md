# Current Task

## Current task

Prepare EXP004–EXP007 for joint human review. All four are PLANNED, independent
Val-only experiments against the frozen Stage C-v0 baseline. Their source
protocols, configs and Val-only `run.sh` entry points are present under
`experiments/stage_c_v1/`.

## Frozen

Stage A/B/C-v0 artifacts, ST-GCN, split, candidate pool, current viewpoint,
decision protocol, and all accepted runtime data. EXP003 is rejected; its
diagnostics are evidence only and are not a baseline or a dependency.

## Do not

- train EXP004–EXP007;
- run Test;
- regenerate Habitat, YOLO, VideoPose3D, Stage A/B/C-v0 or EXP003 artifacts;
- create EXP008 or enter Stage D.
