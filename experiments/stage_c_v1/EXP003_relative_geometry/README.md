# EXP003 — Relative Geometry Representation

## Status

COMPLETED — REJECTED

## Scientific Question

Can explicit candidate-set-relative geometry help the Set Ranker discriminate
between candidates without imposing a preference for farther viewpoints?

## Hypothesis

Stage C errors may partly arise because the current geometry describes each
candidate mostly in local terms, without explicitly encoding its position
relative to the other candidates in the same set. Adding normalized set
statistics and ranks may reduce systematic radius-selection bias.

This does **not** assume that farther views are better.

## Baseline

Frozen Stage C-v0 `SetUtilityRanker` with record-balanced sampling and the
original SmoothL1 plus stay-inclusive listwise ranking loss. The frozen Val
baseline is recorded in `baseline.json`.

## Single Change

Candidate geometry representation only. The architecture, optimizer, loss,
sampler and decision protocol remain unchanged.

## Geometry Schema

The frozen cache has 11 candidate features, in this order:

1. `ego_relative_position_x`
2. `ego_relative_position_y`
3. `ego_relative_position_z`
4. `euclidean_distance_m`
5. `geodesic_distance_m`
6. `sin_relative_azimuth`
7. `cos_relative_azimuth`
8. `path_ratio`
9. `current_radius_m`
10. `candidate_radius_m`
11. `delta_radius_m`

EXP003 appends five candidate-set-relative continuous features:

1. `radius_zscore`: candidate radius standardized within the candidate set;
2. `radius_rank`: normalized rank from nearest (`0`) to farthest (`1`);
3. `geodesic_zscore`: geodesic distance standardized within the set;
4. `geodesic_rank`: normalized rank from shortest (`0`) to longest (`1`);
5. `delta_radius_normalized`: `delta_radius_m` divided by mean candidate radius.

The resulting geometry dimension is 16. Near-zero set standard deviations
produce zero z-scores, and singleton sets produce finite zero/rank features.
Relative features are computed only from the current episode's candidate
geometry set; no utility, label or future perception is used.

## Feature Cache

The independent cache is generated at
`ACTIVEVIEW_DATA_ROOT/datasets/policy_v11_5/stage_c_v1/EXP003_relative_geometry/`
by `build_stage_c_relative_geometry_cache.py`. It transforms the accepted
Stage C-v0 JSONL and does not rerun Habitat, YOLO, VideoPose3D or ST-GCN
perception extraction.

## Frozen Components

- Stage A and Stage B artifacts;
- frozen ST-GCN checkpoint;
- 275-D current feature;
- SetUtilityRanker architecture (with geometry input dimension supplied by the
  feature schema);
- record-balanced sampler, 16 episodes per record;
- optimizer, scheduler and base loss (`lambda_gap = 0`);
- candidate pool, decision rule and Train/Val/Test record split.

## Evaluation and Acceptance

Primary metric: Val mean regret. The preregistered target is at least a 5%
relative improvement from `1.4504976684431725` (approximately `1.3780`).

Secondary metrics are lower C2 rate, lower P90 regret and higher headroom.
Macro-F1 must not drop by more than 0.5 percentage points. Acceptance is
decided after the Val-only run; this README does not authorize Test.

## Run

`run.sh` was run once after explicit authorization. It rebuilt the independent
feature-only cache, trained the Set Ranker and evaluated Val only. No Test
evaluation was run.

## Result

See `result.json` for the compact Val result and `analysis.md` for the
geometry-bias diagnostic. The preregistered 5% mean-regret target was not met,
so EXP003 is recorded as REJECTED. Its diagnostic evidence is retained for
independent follow-up experiments.
