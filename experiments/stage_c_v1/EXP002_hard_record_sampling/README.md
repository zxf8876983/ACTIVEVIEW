# EXP002 — Hard-Record-Aware Sampling

## Status

REJECTED

## Scientific Question

Are Stage C catastrophic viewpoint-selection failures partly caused by
insufficient training exposure to difficult motion records?

## Hypothesis

Increasing training exposure to Train-only high-regret motion records, while
keeping approximate per-epoch training volume unchanged, will reduce Val tail
regret and C2 high-loss errors without materially degrading recognition
Macro-F1.

## Evidence

In the frozen EXP001 Val analysis, the worst 10% of motion records contributed
37.43% of catastrophic failures (approximately 3.74× the uniform share).
This is a motivation for the experiment, not an EXP002 result.

## Baseline

Frozen Stage C-v0 `SetUtilityRanker` with record-balanced sampling and the
original SmoothL1 plus stay-inclusive listwise ranking loss.

## Single Change

Replace record-balanced sampling with Train-only hard-record-aware sampling.
The base loss is restored with `lambda_gap = 0`.

## Hard Record Definition

Rank the 589 Train motion records by mean SafeOracle regret from frozen Stage
C-v0 Train predictions. The top 20% (118 records after deterministic ceiling)
are hard records. Val and Test records are never used to define this set.

## Sampling

- Hard records: 32 Episodes per record per epoch;
- Normal records: 12 Episodes per record per epoch;
- Approximate total exposure: 9,428 samples/epoch versus 9,424 for baseline.

## Frozen

Stage A/B/C-v0 artifacts, feature schema, SetUtilityRanker architecture,
optimizer, scheduler, base loss, seed, epoch limit, batch size, candidate
protocol, record split, decision rule and evaluation metrics.

## Primary Metric

Val P90 regret. The preregistered target is at least a 5% relative reduction
from the frozen baseline value `5.607817676663398` (target approximately
`5.3274`).

## Secondary Metrics

Val Accuracy, Macro-F1, mean/median/P95/P99 regret, positive headroom capture,
C2 rate, top-5% mean regret and top-5% regret threshold.

## Acceptance

ACCEPT requires the P90 regret reduction to be at least 5%, at least one of
mean regret/C2/headroom/top-5% mean regret to improve, and Macro-F1 to drop by
no more than 0.5 percentage points. The criteria are fixed before training.

## Run

Executed on commit `330310f9486ae320b59e12d44e249795c6d903f6`. The run used
Train-only hard-record-aware sampling and evaluated Val only; Test was not
used.

## Result

See `analysis.md`. The primary Val P90 regret criterion was not met, so EXP002
is rejected. No sampler parameters were searched after this result.
