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

Decision: REJECT. Five candidate-set-relative continuous features were appended
to the frozen 11-D geometry and evaluated on Val only. Accuracy/Macro-F1
improved by 0.586/0.683 percentage points; mean regret improved 2.094%, below
the pre-registered 5% target; P90 and headroom improved modestly, while C2
worsened 1.265 points. The diagnostic found a persistent near-radius shortcut
(77.09% closer selections versus 55.01% for SafeOracle). Results remain
evidence for independent follow-up experiments. Test was not used.

## EXP004–EXP007 — Diagnostic experiment preparation

2026-08-29: prepared four independent PLANNED Val-only experiments against the
frozen Stage C-v0 baseline: radius ablation, circular direction features,
current-only Move/Stay gate, and explicit candidate relations. No experiment
was trained and no Test evaluation was run.
