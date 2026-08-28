# EXP001 — Utility-Gap-Aware Ranking Objective

## Experiment ID

`EXP001`

## Scientific Question

Does emphasizing pairwise ordering errors with larger ground-truth utility
gaps reduce harmful viewpoint choices on the held-out Val motion records,
without changing the Stage C-v0 representation, architecture, or sampling
protocol?

## Hypothesis

Some high-regret episodes arise because the baseline objective treats a small
and a large utility-ordering mistake similarly. A single utility-gap-weighted
pairwise ranking term will improve Val mean/p90 regret and positive-headroom
capture while preserving or improving recognition Accuracy and Macro-F1.
This is a falsifiable hypothesis: the pre-registered acceptance criteria
below must be met on Val without inspecting Test.

## Motivation and Baseline

The frozen Stage C-v0 `SetUtilityRanker` uses SmoothL1 utility regression plus
stay-inclusive listwise ranking. Its Test-only failure analysis is not used
for tuning here; it motivates testing whether the objective itself underweights
catastrophic utility gaps. The baseline for this experiment is the same model,
features, sampler, optimizer family and protocol with the new term disabled.

## Single Core Change

Add exactly one stay-inclusive utility-gap-weighted pairwise ranking term to
the existing Set Ranker loss. No architecture, feature, candidate protocol,
sampler, optimizer, upstream artifact, or evaluation split is changed.

## Loss Definition

Stay is represented as utility and score zero. For every valid ordered pair
`i,j` with `u_i > u_j`, define

```text
gap_ij = u_i - u_j
w_ij = min(gap_ij / tau_gap, max_weight)
L_gap = sum_ij w_ij * softplus(-(s_i - s_j)) / sum_ij w_ij
L_total = L_existing + lambda_gap * L_gap
```

Padding candidates are excluded by the existing candidate mask. The planned
parameters are `lambda_gap=1.0`, `tau_gap=1.0`, and `max_weight=10.0`.
The implementation normalizes over all valid ordered pairs in a training
batch (global pair normalization), rather than averaging independently
normalized losses per Episode. This intentionally gives batches with larger
total utility-gap mass more influence; per-Episode normalization is deferred to
a separate ablation and is not part of EXP001.

## Frozen Components

- Accepted Stage A Episodes and geometry/candidate protocol
- Accepted Stage B utility labels and oracle definitions
- Accepted Stage C feature cache and normalization statistics
- Frozen ST-GCN checkpoint and label mapping
- Train/Val/Test motion-record split (`589/197/194`)
- Set Utility Ranker architecture, record-balanced sampler, optimizer family,
  learning-rate schedule and data preprocessing
- Test lock: no Test evaluation before a separately authorized final run

## Metrics

On Val, report Accuracy, Macro-F1, mean/median/p90 regret, positive-headroom
capture, Stay rate, SafeOracle gap, utility-gap quartile metrics, C2
wrong-candidate high-loss rate, and catastrophic top-5% regret rate. Record
the baseline and EXP001 values using identical evaluation code.

## Acceptance Criteria (pre-registered)

Relative to the frozen Stage C-v0 baseline, accept only when all of the
following are true on Val:

1. Large-gap utility-quartile mean regret decreases by at least 5%.
2. At least one additional harmful-ranking diagnostic improves: C2
   wrong-candidate high-loss rate decreases, Val p90 regret decreases, or
   positive-headroom capture increases.
3. Val Macro-F1 decreases by no more than 0.5 percentage points.
4. No frozen-artifact, split, leakage, or provenance validator check fails.
5. `test_used` remains `false` throughout the experiment.

## Rejection Criteria (pre-registered)

Reject when the large-gap mean regret does not decrease by at least 5% and
there is no consistent improvement in the other harmful-ranking diagnostics.

## Inconclusive Criteria (pre-registered)

Mark `INCONCLUSIVE` if large-gap harmful errors improve but Macro-F1 decreases
by more than 0.5 percentage points, or if the key diagnostics give conflicting
signals. An inconclusive result must not trigger an automatic follow-up
experiment.

## Forbidden Changes

- Stage A/B or accepted Stage C-v0 artifacts
- Model architecture, feature schema, sampler, optimizer or protocol changes
- Test evaluation before explicit final authorization
- Starting this experiment or creating EXP002 without human approval
