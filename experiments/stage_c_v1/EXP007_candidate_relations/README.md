# EXP007 — Explicit Candidate Relations

## Status

COMPLETED — PENDING USER REVIEW

## Question

Does the ranker need explicit candidate-to-candidate geometry summaries in
addition to its existing Transformer interaction?

## Hypothesis

Simple normalized relation and density features will make candidate-set
structure easier to learn without changing the Set Ranker architecture.

## Evidence

The frozen Set Ranker already models interactions, but the Stage C-v0 geometry
does not expose explicit set-distance or density summaries.

## Baseline

Frozen Stage C-v0 Set Ranker and Val metrics in `baseline.json`.

## Single Change

Append five relation features to the frozen 11-D geometry: normalized radius and
geodesic deltas from the set mean, nearest and mean geometry-space distances,
and candidate density. Geometry-space distances use normalized radius,
geodesic and wrapped azimuth. The resulting geometry is 16-D.

## Frozen

Stage A/B/C-v0, ST-GCN, current feature, Set Ranker/Transformer, record-balanced
sampler, loss, optimizer, scheduler, seed, candidate pool, decision rule and
split.

## Primary Metric

Val mean regret; target relative improvement of at least 5% over the frozen
baseline.

## Secondary Metrics

P90 regret, C2 rate, positive headroom capture, Macro-F1 and candidate-set
difficulty diagnostics.

## Acceptance

Primary target plus at least one secondary improvement; Macro-F1 drop no more
than 0.5 percentage points. Acceptance is a later human decision.

## Run

`run.sh` completed Train→Val only. No Test evaluation was run. Full runtime
outputs are under `${ACTIVEVIEW_DATA_ROOT}/experiments/stage_c_v1/EXP007_candidate_relations/`.

## Result

See `result.json` for the compact Val result. Mean regret was 1.4505859 and
the radius diagnostic still selected closer views in 79.16% of moves
(SafeOracle: 55.01%). No Test evaluation was run.
