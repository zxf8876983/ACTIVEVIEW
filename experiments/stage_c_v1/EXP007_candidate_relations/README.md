# EXP007 — Explicit Candidate Relations

## Status

PLANNED

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

Pending human approval. `run.sh` is Val-only and has not been executed.

## Result

Pending. No training or Test evaluation has been run.
