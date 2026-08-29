# Current Task

## Current experiment

EXP003 — relative geometry representation

Status: **PLANNED**; human review is required before training.

## Single Change

Append five candidate-set-relative geometry features to the frozen 11-D
candidate geometry, producing an independent 16-D feature cache.

## Frozen

Stage A/B/C-v0 artifacts, ST-GCN, current feature, Set Ranker architecture,
record-balanced sampler, loss, optimizer, scheduler, split, candidate pool and
decision rule.

## Primary

Val mean regret; target at least 5% relative improvement from 1.4504976684.

## Artifacts

Feature cache: `ACTIVEVIEW_DATA_ROOT/datasets/policy_v11_5/stage_c_v1/EXP003_relative_geometry/`.

## Next

Human review before EXP003 training.

## Do not

- train EXP003;
- run Test;
- modify Stage A/B/C-v0 artifacts;
- create EXP004.
