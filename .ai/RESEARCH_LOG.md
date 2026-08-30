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

## Stage D geometry-semantics audit

2026-08-30: post-run review identified that Stage D `second_step_geometry()`
had populated the Stage C sixth/seventh features using an s1-camera-frame
displacement bearing. This is not Stage A's radial
`candidate_azimuth_deg - current_azimuth_deg` definition. The correction now
loads the existing semantic-region-v2 candidate metadata, computes the radial
azimuth difference with the exact [-180°, 180°) wrap, and retains s1 rotation
only for egocentric XYZ. A focused regression test covers 90° semantics and
wrapping. No Stage A/B/C-v0 or perception artifact changed. EXP014/EXP015
runtime outputs were generated pre-fix and are preserved for traceability; no
Test was used.

## Stage D corrected geometry rerun

2026-08-30: archived the pre-fix EXP014 runtime, rebuilt the Stage D cache
using the existing semantic-region-v2 radial azimuth metadata, and reran
EXP014 Train→Val plus EXP015 Val-only analysis. Corrected EXP014 selected epoch
11 and achieved Accuracy 0.658254, Macro-F1 0.610153, mean regret 1.422463,
P90 regret 5.515663 and aggregate positive headroom capture 0.783313. Relative
to frozen v0, mean regret improved 1.93% and Accuracy improved 0.915 points;
the recorded reject thresholds were met, so EXP014 is REJECT. EXP015's
fixed-first second-step oracle remained Accuracy 0.771502 / Macro-F1 0.725081 /
mean regret 0.586204 / P90 1.699901 / headroom 0.890887; corrected EXP014
second-step action match was 46.33% and move-only exact hit 17.07%. Both runs
used Val only (`test_used=false`); no Stage A/B/C-v0, perception or ST-GCN
artifact changed.

## EXP016 preparation — second-step error decomposition

2026-08-30: prepared a no-training, Val-only analysis to isolate corrected
EXP014 second-step Stay/Move gate errors from p2/p3 candidate-ranking errors.
All variants preserve the frozen Stage C-v0 first action and use true U2 only
for offline oracle branches. Added policy primitives, a Val-only CLI, an
experiment record and focused synthetic tests. The real Val analysis has not
been executed; Test remains locked.

## EXP019 — Executed-Candidate-Aware Second-Step Gate

2026-08-31: trained one fixed 12-D binary gate for exactly 30 Train epochs,
supervised by `1[true_U2(c_hat)>0]` with frozen EXP014 candidate ranking, then
evaluated once on Val. EXP019 achieved Accuracy 0.656681, Macro-F1 0.607936,
mean regret 1.429851, P90 regret 5.532555 and headroom capture 0.780325,
slightly worse than EXP014 (0.658254 / 0.610153 / 1.422463). The executed-
candidate oracle remains 0.743119 Accuracy / 0.761339 mean regret. The gate
changed 984 Stay→Move and 253 Move→Stay decisions, with zero candidate
identity mismatches. Decision: **INCONCLUSIVE**; no Test was read or used,
and no frozen upstream artifact was changed.

2026-08-30: corrected the EXP016 preparation after protocol review. Oracle
candidate selection now exactly reproduces frozen EXP015 `np.argmax` behavior,
including cached-order ties; learned selection remains utility/geodesic/ID
ordered. The Val analyzer now fails closed unless Stage B and frozen-v0 IDs
match exactly and the Stage D cache plus EXP014 predictions equal the frozen-v0
Move subset. No real Val analysis or Test access was performed.

2026-08-30: executed the authorized EXP016 Val-only second-step error
decomposition. The 13,987-episode run reproduced frozen EXP014 and
Fixed-first Oracle references exactly. OracleGate + LearnedCandidate reached
0.720026 Accuracy / 0.670190 Macro-F1 / 0.969138 mean regret, while LearnedGate
+ OracleCandidate reached 0.683778 / 0.638848 / 1.235322; the fixed-first
oracle reached 0.771502 / 0.725081 / 0.586204. Gate correction recovered
54.55% of the joint Accuracy gap and 54.21% of the joint mean-regret gap;
candidate correction recovered 22.54% and 22.38%. EXP016 is analysis-only
**INCONCLUSIVE**; Test was not read and no training or upstream artifact was
modified.

## EXP017 preparation — second-step gate calibration audit

2026-08-30: prepared EXP017 as a no-training diagnostic. The implementation
fits one strict scalar `gate_score > tau` threshold from frozen EXP014 Train
predictions using balanced accuracy, with deterministic Move-F1/zero-distance/
larger-threshold tie-breaking, then applies the frozen threshold once to Val.
The learned p2/p3 candidate ordering and frozen Stage C-v0 first decision are
unchanged. Calibration artifacts are written before Val evaluation, explicit
Test input is rejected, and candidate identity is audited across tau variants.
Only synthetic unit tests and static checks are authorized at this stage; real
EXP017 Train calibration and Val evaluation have not been executed.

## EXP017 — second-step gate calibration result

2026-08-31: executed the authorized EXP017 no-training Train→Val audit using
the corrected frozen EXP014 checkpoint. Train selected the strict threshold
`tau=-0.08218251913785934` from 29,133 second-step episodes. On 13,987 Val
episodes, the calibrated policy reached Accuracy 0.650962 / Macro-F1 0.598102 /
mean regret 1.477153 / P90 regret 5.605605 / headroom 0.777146, compared with
EXP014 tau=0 at 0.658254 / 0.610153 / 1.422463 / 5.515663 / 0.783313. The
threshold changed 2,838 Stay decisions to Move and none in the reverse
direction; learned candidate identity was unchanged in all 2,356 episodes
where both policies moved. Gate balanced accuracy improved on Val from
0.571825 to 0.609873, but trajectory performance worsened. EXP017 is
**REJECTED** as a deployable global-threshold intervention. Test was not read;
no training, perception regeneration, Habitat rendering or upstream artifact
modification was performed.

## EXP018 — executed-candidate gate alignment audit

2026-08-31: executed the authorized EXP018 Val-only offline audit without
training or new perception. The Stage C-v0 first decision/p1 and corrected
EXP014 learned p2/p3 ranking were frozen. Among 9,742 v0-Move episodes,
`y_any=max(true U2)>0` was positive for 5,477 while the true utility of the
frozen learned candidate (`y_exec`) was positive for 4,281. The ranking-induced
mismatch count was 1,196 (21.84% of any-positive episodes); the impossible
`y_any=Stay,y_exec=Move` case was zero. ExecutedCandidateOracleGate + learned
candidate reached Accuracy 0.743119 / Macro-F1 0.693231 / mean regret 0.761339,
versus AnyPositiveOracleGate 0.720026 / 0.969138 mean regret and EXP014
0.658254 / 1.422463. Of EXP017's 2,838 additional Stay→Move decisions, 1,497
were non-positive for the candidate actually selected by EXP014. EXP018 is an
analysis-only **INCONCLUSIVE** diagnostic; Test was not read and no upstream
artifact was modified.

## EXP020 — Frozen EXP014 Contextual-Latent Executed Gate

2026-08-31: trained a fixed 129-D binary gate using the frozen EXP014
contextual candidate token (extracted before its utility head) plus frozen
predicted utility. After 30 fixed Train epochs, Val Accuracy was 0.661757,
Macro-F1 0.612956, mean regret 1.453868, P90 regret 5.635810 and headroom
0.778625. Gate ROC-AUC/PR-AUC were 0.641344/0.554610. Accuracy rose modestly
over EXP014 but regret worsened; decision **INCONCLUSIVE**. Test and all
upstream perception/artifacts remained untouched.

## EXP021 — Offline Contextual-Bandit Joint Second-Step Policy

2026-08-31: trained a new contextual scorer for the full-information
one-step action set `{Stay,p2,p3}` by maximizing expected Train utility under
`softmax([0,q2,q3])`, with Stay fixed at zero. After 30 fixed epochs, the
policy selected Stay for all 9,742 Val v0-Move episodes, yielding Accuracy
0.649103, Macro-F1 0.598042, mean regret 1.450498, P90 regret 5.607818 and
headroom 0.777965. The formulation was **REJECTED** as a useful policy under
the fixed protocol; no Test or upstream artifact was used.

## EXP022 — Executed-Candidate Utility Regression Gate

2026-08-31: trained the fixed 129-D EXP020 contextual head with raw
executed-candidate `true_U2(c_hat)` targets for 30 Train epochs using default
SmoothL1 loss, then applied the strict predicted-utility-positive gate once on
Val. EXP022 reached Accuracy 0.659898, Macro-F1 0.611687, mean regret
1.416495, P90 regret 5.494913 and headroom 0.782352, versus EXP014's
0.658254 / 1.422463. It recovered 1.94% of the executed-candidate oracle
Accuracy gap and 0.90% of its mean-regret gap, with zero candidate identity
mismatches. Decision: **ACCEPTED as research-direction evidence**, not final
policy acceptance. Test was not read and frozen upstream artifacts were not
modified.

## EXP023 — Supervised-Warm-Started Contextual Bandit

2026-08-31: trained a fixed contextual scorer for 20 Train-only epochs of
candidate-U2 SmoothL1 warm-start followed by 10 Train-only epochs of fixed
full-information expected-reward optimization with entropy bonus 0.001. The
warm start avoided EXP021's all-Stay collapse. EXP023 selected Stay/p2/p3 on
Val at 6903/1730/1109 episodes and reached Accuracy 0.660470, Macro-F1
0.608566, mean regret 1.374664, P90 regret 5.294162 and headroom 0.786731.
It exceeded EXP014 on Accuracy by 0.002216 and reduced mean regret by
0.047799, recovering 1.96% / 5.72% of the fixed-first oracle gaps. Decision:
**ACCEPTED as research-direction evidence**, not final-policy acceptance. Test
was not read, no Val tuning occurred, and no EXP024 was started.

## RGB Observation Dataset V1

2026-08-31: generated a separate raw RGB observation dataset aligned one-to-one
with the canonical HM3D-train skeleton records. The source skeleton root was
read-only; RGB output was written under `/home/zxf/MG08/robot/ActiveView/`.
Each of 82,320 records contains 32 `uint8` RGB views at `[256,256,3]`, all
restored from fixed motion frame index 15 and the saved skeleton camera states.
The 21-scene output contains 2,634,240 viewpoints and occupies
214,924,343,585 bytes (200.16 GiB); full audit found 0 missing, 0 extra and 0
invalid records. Generation used 16 Habitat workers per scene after explicit
authorization. No YOLO, VideoPose3D, ST-GCN or skeleton regeneration was run,
and no policy Test evaluation or embedding extraction was started.
