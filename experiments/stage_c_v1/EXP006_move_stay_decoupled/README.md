# EXP006 — Move/Stay Decoupled

## Status

PLANNED

## Question

Is the single utility output entangling the decision to move with the choice
of which candidate to visit?

## Hypothesis

A current-only Move/Stay gate will reduce missed valuable moves while the
unchanged candidate ranker chooses the destination when moving.

## Evidence

EXP003 moved in 71.75% of Val Episodes versus 85.35% for SafeOracle. This
motivates a gate diagnostic, not a change to the frozen input representation.

## Baseline

Frozen Stage C-v0 Set Ranker and Val metrics in `baseline.json`.

## Single Change

Add a current-context-only Move/Stay head to the Set Ranker. The gate is trained
with SafeOracle-not-Stay targets using `lambda_move=1.0` BCE and uses a fixed
sigmoid threshold of 0.5 at inference. Candidate utility ranking is unchanged;
the resulting input geometry remains 11-D.

## Frozen

Stage A/B/C-v0, ST-GCN, current feature, 11-D geometry, record-balanced
sampler, base loss, optimizer, scheduler, seed, candidate pool, decision rule
and split. The SafeOracle target is supervision only, never an input feature.

## Primary Metric

Val mean regret; target relative improvement of at least 5% over the frozen
baseline.

## Secondary Metrics

P90 regret, C2 rate, positive headroom capture, Macro-F1, missed-move rate,
unnecessary-move rate and Move rate.

## Acceptance

Primary target plus evidence that missed valuable moves decrease and Move rate
moves toward SafeOracle; Macro-F1 drop no more than 0.5 percentage points.
Acceptance is a later human decision.

## Run

Pending human approval. `run.sh` is Val-only and has not been executed.

## Result

Pending. No training or Test evaluation has been run.
