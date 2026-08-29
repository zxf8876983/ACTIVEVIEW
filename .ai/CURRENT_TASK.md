# Current Task

## Current task

EXP004–EXP007 Train→Val runs are complete and recorded as rejected diagnostic
directions. Stage C-v2 EXP008–EXP010 have now completed their authorized
Train→Val runs and are pending human scientific review. Test remains locked.

## Frozen

Stage A/B/C-v0 artifacts, ST-GCN, split, candidate pool, current viewpoint,
decision protocol, and all accepted runtime data. EXP003 is rejected; its
diagnostics are evidence only and are not a baseline or a dependency.

## Do not

- run Test;
- regenerate Habitat, YOLO, VideoPose3D, Stage A/B/C-v0 or EXP003 artifacts;
- run Test for EXP008, EXP009 or EXP010;
- start a new v2 experiment or add body-yaw features without explicit
  authorization;
- enter Stage D.

## Completed v2 run summary

The shared cache was built from frozen Stage A/B/C-v0 artifacts only. Runtime
outputs are under `ACTIVEVIEW_DATA_ROOT/experiments/stage_c_v2/`. EXP008,
EXP009 and EXP010 selected epochs 26, 22 and 54 respectively. Their Val
metrics and the frozen v0 comparison are recorded in the experiment READMEs and
`experiments/stage_c_v2/registry.csv`. The preregistered limitation is that
the current skeleton representation is body-yaw canonicalized and therefore
does not preserve explicit body-to-candidate directional alignment.
