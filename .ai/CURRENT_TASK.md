# Current Task

## Stage D preparation

Stage C-v0 one-shot ranking is frozen. EXP011–EXP013 diagnostics indicate
moderate online utility predictability but strong Top-K proposal coverage, so
the next approved preparation is sequential active view selection.

This change prepares EXP014 (two-step sequential policy) and EXP015
(fixed-first budget/oracle analysis). Both remain **PLANNED** pending human
review. No formal experiment has been run.

## Frozen

Stage A/B/C-v0 artifacts, ST-GCN checkpoint, motion split 589/197/194,
candidate pool, current-view protocol, Stage C-v0 proposal ordering and all
perception data remain frozen.

## Protocol boundaries

- Train and Val only; Test is locked.
- No Habitat rendering, RGB/depth, YOLO, VideoPose3D or ST-GCN retraining.
- No Stage C-v0 retraining, loss/sampler/threshold changes, Top-K sweep,
  exploration or body-yaw feature.
- EXP014 may use only visited s1 perception; unvisited p2/p3 perception is
  never a policy input.
- EXP015 performs no training and fails clearly if EXP014 Val output is absent.

## Preparation status

- [x] Added frozen-v0 Train/Val proposal inference helper.
- [x] Added navigation-only pairwise viewpoint geodesic builder.
- [x] Added Stage D second-step cache schema, s1 frozen ST-GCN reconstruction,
      s1-relative 11-D geometry and U2 supervision construction.
- [x] Added SequentialObservationRanker, Val trajectory evaluator and
      Fixed-first Second-Step Oracle analysis.
- [x] Added EXP014/EXP015 README, config, run scripts and registry entries.
- [x] Added focused Stage D unit tests and compile check.
- [ ] Await human review before any execution.

## Current state

Preparation is complete. No training, Val evaluation, Test evaluation or data
generation was performed in this task.
