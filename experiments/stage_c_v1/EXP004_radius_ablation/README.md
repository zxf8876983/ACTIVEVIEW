# EXP004 — Radius Ablation

## Status

PLANNED

## Question

Does the Set Ranker over-rely on explicit candidate-radius cues as a utility
shortcut?

## Hypothesis

Removing direct radius features will reduce the learned preference for nearer
viewpoints and make direction/geodesic information more useful.

## Evidence

EXP003 showed a strong nearer-is-better bias (77.09% closer selections versus
55.01% for SafeOracle). This is diagnostic evidence only; EXP003 is not the
baseline for this experiment.

## Baseline

Frozen Stage C-v0 Set Ranker and Val metrics in `baseline.json`.

## Single Change

Transform the frozen 11-D geometry to 10-D by removing current radius,
candidate radius, delta radius, and radius-derived EXP003 features. Keep the
remaining egocentric, distance, azimuth, path-ratio and geodesic features,
including geodesic z-score/rank.

## Frozen

Stage A/B/C-v0, ST-GCN, current feature, Set Ranker, record-balanced sampler,
loss, optimizer, scheduler, seed, candidate pool, decision rule and split.

## Primary Metric

Val mean regret; target relative improvement of at least 5% over the frozen
baseline.

## Secondary Metrics

P90 regret, C2 rate, positive headroom capture, Macro-F1 and radius-bias
diagnostics.

## Acceptance

Primary target plus at least one secondary improvement; Macro-F1 drop no more
than 0.5 percentage points. Acceptance is a later human decision.

## Run

Pending human approval. `run.sh` is Val-only and has not been executed.

## Result

Pending. No training or Test evaluation has been run.
