# EXP005 — Direction-Enhanced Geometry

## Status

PLANNED

## Question

Is candidate direction insufficiently explicit relative to the candidate set?

## Hypothesis

Circular, candidate-set-relative azimuth features will help the ranker resolve
directional alternatives without changing the radius representation.

## Evidence

EXP003 showed a modest azimuth shift and persistent radial bias. This experiment
tests directional representation independently from radius ablation.

## Baseline

Frozen Stage C-v0 Set Ranker and Val metrics in `baseline.json`.

## Single Change

Append four circular-safe features to the frozen 11-D geometry: azimuth rank,
deviation from the circular set center, nearest angular-neighbor distance and
angular local density. The resulting geometry is 15-D.

## Frozen

Stage A/B/C-v0, ST-GCN, current feature, original radius features, Set Ranker,
record-balanced sampler, loss, optimizer, scheduler, seed, candidate pool,
decision rule and split.

## Primary Metric

Val mean regret; target relative improvement of at least 5% over the frozen
baseline.

## Secondary Metrics

P90 regret, C2 rate, positive headroom capture, Macro-F1 and azimuth-selection
diagnostics.

## Acceptance

Primary target plus at least one secondary improvement; Macro-F1 drop no more
than 0.5 percentage points. Acceptance is a later human decision.

## Run

Pending human approval. `run.sh` is Val-only and has not been executed.

## Result

Pending. No training or Test evaluation has been run.
