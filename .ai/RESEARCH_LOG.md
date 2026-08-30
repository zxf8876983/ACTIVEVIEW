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

2026-08-29: completed four independent Train→Val experiments against the frozen
Stage C-v0 baseline: radius ablation, circular direction features, current-only
Move/Stay gate, and explicit candidate relations. EXP004–EXP007 results are
recorded as rejected diagnostic directions; no Test evaluation was run.

## Stage C-v2 architecture preparation

2026-08-29: geometry, loss and sampling changes plateaued around 64–65% Val
Accuracy. Prepared three independent architecture candidates: joint-aware
frozen ST-GCN tokens (EXP008), candidate-conditioned joint attention (EXP009),
and direct current-skeleton policy Transformer (EXP010). Before execution, the
following limitation was preregistered: the current skeleton representation is
body-yaw canonicalized and therefore does not preserve explicit body-to-
candidate directional alignment.

## EXP008–EXP010 — Stage C-v2 Val runs

2026-08-30: all three authorized experiments completed Train→Val using the
shared cache built from frozen Stage A/B/C-v0 artifacts. No Test, Habitat, RGB,
YOLO or VideoPose3D processing was performed. EXP008 achieved Accuracy 0.644813,
Macro-F1 0.598766, mean regret 1.474656, P90 regret 5.633397, headroom 0.770660
and C2 0.332952 (epoch 26). EXP009 achieved 0.647530, 0.597071, 1.502271,
5.819350, 0.759083 and 0.308572 (epoch 22). EXP010 achieved 0.651677,
0.598782, 1.458965, 5.660032, 0.784780 and 0.326875 (epoch 54). Results are
recorded as rejected diagnostic directions; no automatic v2 acceptance
decision was made.

## Stage C-v3 predictability diagnostics preparation

2026-08-30: the Stage C-v2 architecture probes did not break the 64–65%
Train-to-Val plateau. Prepared three read-only diagnostics for the next
scientific question: whether the remaining gap is caused by future-perception
information unavailable at decision time. EXP011 is a diagnostic-only
future-perception teacher, EXP012 is a Train-reference/Val-query utility
predictability audit, and EXP013 is a frozen-v0 Top-K reachability audit.
EXP011 remained on HOLD until its schema was corrected; EXP012 and EXP013
remained PLANNED until execution. No Test, Habitat or perception rerun was
performed.

## EXP011 schema correction — HOLD

2026-08-30: review identified that candidate `logp_true` is GT-dependent and
is a direct ingredient of the Stage B utility target. It was removed from the
future-perception teacher input; EXP011 now uses only a 16-D predicted-label
one-hot and 1-D entropy (17-D total). No cache, training or evaluation was
run; EXP011 remains on HOLD pending review.

## EXP011–EXP013 — Stage C-v3 predictability diagnostics

2026-08-30: authorized Train→Val / read-only Val diagnostics completed without
Test. EXP011 used the corrected 17-D future input (predicted-label one-hot plus
entropy) and selected epoch 41: Accuracy 0.652106, Macro-F1 0.619947, mean
regret 1.649950, P90 6.534039, headroom 0.767026. Classification improved over
v0, but regret and headroom did not. EXP012 found modest predictability gains
from adding legal current state to geometry (sign agreement 0.6362→0.6740;
MAE 4.27625→4.17351) while disagreement remained high. EXP013 found Top-5
coverage of 87.16% CandidateOracle hit, 86.30% SafeOracle move hit and
0.04842 mean move-only Top-K regret. These are diagnostic results pending
human review; no Stage A/B/C-v0 artifact changed and no Test evaluation was
performed.

## Stage D preparation

2026-08-30: EXP013 Top-3/Top-5 proposal coverage motivated reinterpreting the
frozen Stage C-v0 model as a first-stage proposal mechanism. Prepared EXP014
two-step sequential policy and EXP015 fixed-first sequential budget/oracle
analysis. The preparation adds only navigation-only pairwise geodesic cache
support, frozen s1 ST-GCN feature reconstruction and a new second-step head;
it does not run training/evaluation or alter Stage A/B/C-v0 artifacts. Test
remains locked pending human review.

## EXP014–EXP015 — Stage D Val-only execution

2026-08-30: executed the authorized Stage D study without changing frozen
Stage A/B/C-v0 artifacts. EXP014 trained the two-step sequential ranker on
Train and selected epoch 24 by complete Val trajectory Macro-F1. On 13,987 Val
episodes it achieved Accuracy 0.664331, Macro-F1 0.615151, mean regret 1.397287,
P90 regret 5.403128 and aggregate positive headroom capture 0.783344, versus
frozen v0 values 0.649103, 0.598042, 1.450498, 5.607818 and 0.777965. Mean
regret improved 3.67%, below the preregistered strong-success threshold; the
controlled decision is **INCONCLUSIVE**. The policy averaged 0.916 moves and
2.562 m trajectory cost.

EXP015 then performed the fixed-first second-step budget/oracle analysis on
the same Val split. Of 4,245 frozen-v0 Stay episodes, 2,926 (68.93%) would
move under SafeOracle. Among 9,742 v0-move episodes, EXP014 matched the
second-step oracle action 46.73% of the time and had 22.42% move-only exact
candidate hit. The fixed-first second-step oracle reached Accuracy 0.771502,
Macro-F1 0.725081, mean regret 0.586204 and P90 regret 1.699901. EXP015 is an
analysis-only **INCONCLUSIVE** result. Both experiments used `test_used=false`;
no Test, Habitat, RGB, YOLO, VideoPose3D or ST-GCN retraining was performed.
