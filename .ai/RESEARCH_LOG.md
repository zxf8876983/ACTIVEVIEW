# Research Log

## Stage C-v0

Set Ranker established the viability of current-conditioned viewpoint utility
prediction. NoMove, learned policies and SafeOracle remain offline diagnostics.

## Stage C-v0 failure analysis

Wrong-candidate high-loss errors and hard motion records dominate the long tail;
body-orientation evidence was weak/inconclusive. This supports investigating
gap-aware and hard-example objectives before changing representation.

## EXP001 — Gap-Aware Ranking

Status: REJECTED. The hypothesis, frozen baseline and single loss change are
recorded in `experiments/stage_c_v1/EXP001_gap_aware_ranking/`; its Val-only
run did not meet the preregistered target. No Test evaluation was used.

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

Decision: REJECT. The Val-only run with 32 Episodes/record for hard records
and 12 for normal records did not improve P90 regret (5.6153066 versus the
5.6078177 baseline), mean regret, headroom or C2. Macro-F1 decreased by 0.44
percentage points. Test was not used.

## EXP003 — Relative Geometry Representation

Status: PLANNED. Five candidate-set-relative continuous features are appended
to the frozen 11-D geometry: radius/geodesic z-scores and ranks, plus
normalized delta radius. The independent 16-D cache was built by transforming
the accepted Stage C-v0 feature JSONL; no Habitat, perception, training or Test
evaluation was run. Human review is required before training.
