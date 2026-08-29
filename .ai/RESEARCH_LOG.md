# Research Log

## Stage C-v0

Set Ranker established the viability of current-conditioned viewpoint utility
prediction. NoMove, learned policies and SafeOracle remain offline diagnostics.

## Stage C-v0 failure analysis

Wrong-candidate high-loss errors and hard motion records dominate the long tail;
body-orientation evidence was weak/inconclusive. This supports investigating
gap-aware and hard-example objectives before changing representation.

## EXP001 — Gap-Aware Ranking

Status: PLANNED. The hypothesis, frozen baseline and single loss change are
recorded in `experiments/stage_c_v1/EXP001_gap_aware_ranking/`. No training or
Test evaluation has been run.

## Workflow simplification

2026-08-29: the engineering-heavy experiment lifecycle was simplified to
lightweight README/config/run/result/analysis records. Accepted scientific
artifacts remain unchanged.

## EXP001 — Gap-Aware Ranking

Decision: REJECT. Large-gap mean regret improved only 0.32%, far below the
pre-registered 5%; mean/P90 regret, C2 and headroom did not improve. The tested
utility-gap weighting is therefore insufficient to explain or fix the main
Stage C failure mode.

## EXP002 — Hard-Record-Aware Sampling

Status: PLANNED. The frozen Stage C-v0 Train inference identified 118 hard
records among 589 (top 20% by mean SafeOracle regret). Sampling is prepared
with 32 Episodes/record for hard records and 12 for normal records, while
keeping approximate epoch volume unchanged. No EXP002 training has run.
