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

## EXP028 — Oracle Action Predictability / Representation Sufficiency Audit

2026-08-31: completed the authorized Val-only diagnostic on the frozen
Stage-D second-step eligible population (29,133 Train / 9,742 Val). The
observable vector concatenated Train-statistics-normalized s0/s1/delta and
candidate geometry with Train-statistics-normalized pooled visited s0/s1
EXP025 spatial RGB. A cosine NN index was built from Train only; Val was never
added to the index or used for normalization. EXP027 three-way imitation
accuracy was 0.4730 in the smallest top-1-margin bin and 0.5623 for margin
>=2.0 (0.5376 for >=1.0). NN 3-way agreement was 0.454527/0.449497/0.451139/
0.444570 for k=1/5/10/25, with 25-NN mean three-way entropy 0.870910.
Quantized same-context cross-motion oracle-action switching was 0.551969 on
Val. Decision: **INCONCLUSIVE** pending human Case A/B/C review. This only
assesses predictability under the frozen legal representation and held-out
motion protocol; it does not claim intrinsic unpredictability. Test was not
read, no policy was trained, and no future-candidate observation was used.

## EXP027 — Spatial-RGB Oracle Behavior Cloning

2026-08-31: trained an unweighted three-action CrossEntropy behavior-cloning
policy for 30 fixed Train-only epochs using legal Stage-D skeleton/delta and
visited s0/s1 EXP025 spatial RGB features. Train labels exactly matched the
frozen Fixed-first Second-Step Oracle; no future candidate RGB/depth/skeleton
was accessed and the Stage C-v0 first action remained frozen. On 9,742 Val
second-step episodes, EXP027 predicted Stay/p2/p3 at 4,411/2,923/2,408,
three-way imitation accuracy 0.493841 and binary Move/Stay accuracy 0.644837.
Trajectory Accuracy was 0.657039, Macro-F1 0.608384, mean regret 1.486694,
P90 regret 5.751852 and headroom 0.769306, below EXP014 (0.658254 / 1.422463)
and EXP023 (0.660470 / 1.374664). Decision: **INCONCLUSIVE** negative
diagnostic; direct unweighted action cloning did not improve trajectory
performance under this representation. Test remained locked and no upstream,
RGB or perception artifact was modified.

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

## EXP024 — DINOv2 RGB Context Utility Regression Pilot

2026-08-31: completed the authorized EXP024 Train→Val pilot using only
already visited Stage-D `s0` and `s1` RGB observations. A deduplicated cache of
77,750 observations (58,266 Train and 19,484 Val) was extracted with frozen
Hugging Face `facebook/dinov2-base` ViT-B/14 global CLS embeddings (768-D,
float16); cache extraction took 1,354.320 seconds and occupied 141,402,204
bytes. The RGB projector and 513-D SmoothL1 raw executed-utility regression
head were trained for 30 fixed Train epochs; final Train loss was 1.456738.

On 13,987 Val episodes, EXP024 reached Accuracy 0.657682, Macro-F1 0.609503,
mean regret 1.414353, P90 regret 5.431195 and headroom 0.782434. Utility
regression MAE/RMSE were 2.896349/4.314998, Pearson 0.428491 and Spearman
0.252335. Compared with frozen EXP022 (Accuracy 0.659898, mean regret
1.416495), EXP024 changed Accuracy by -0.002216 and mean regret by -0.002142;
the Accuracy recovery fraction versus the executed-candidate oracle was
-0.006740 and regret recovery was 0.012266. Candidate identity mismatch was
zero. EXP024 is **INCONCLUSIVE** as evidence for a useful global-CLS RGB
policy improvement: regret improved slightly while Accuracy declined, so this
does not support escalating automatically to spatial RGB features. Test was
not read, no future-candidate RGB was accessed, and no upstream or RGB dataset
artifact was modified.

## EXP025 — DINOv2 Spatial RGB Utility Regression

2026-08-31: completed the authorized fixed Train-to-Val pilot using frozen
`facebook/dinov2-base` patch tokens from only visited Stage-D `s0`/`s1` RGB
observations. The deduplicated cache contains 77,750 observations (58,266
Train; 19,484 Val), with no future-candidate RGB. EXP025 reached Val Accuracy
0.659898, Macro-F1 0.607331, mean regret 1.378650, P90 5.330633 and headroom
0.784692; utility MAE/RMSE were 2.839231/4.311770, Pearson 0.439911 and
Spearman 0.266806. Relative to EXP024, mean regret improved by 0.035703 but
Macro-F1 declined by 0.002172. Decision: **INCONCLUSIVE**. Candidate mismatch
was zero; Test and all upstream artifacts remained untouched.

## EXP026 — Spatial RGB-D Utility Regression

2026-08-31: completed the authorized fixed Train-to-Val pilot with Habitat
metric depth rendered only for the same 77,750 visited `s0`/`s1` observations at
frame 15. Depth was pooled to `[16,4]` float16 features using 16 workers and
four-camera batching; no p2/p3 depth was rendered or loaded. EXP026 reached Val
Accuracy 0.657325, Macro-F1 0.606847, mean regret 1.407385, P90 5.423377 and
headroom 0.783227; utility MAE/RMSE were 2.838240/4.305187, Pearson 0.428920
and Spearman 0.248698. Relative to EXP025, Accuracy declined by 0.002574 and
mean regret worsened by 0.0287345. Decision: **INCONCLUSIVE**; depth adds no
evidence beyond spatial RGB in this pilot. Candidate mismatch was zero. Test
remained locked and no Stage A/B/C, skeleton, RGB, perception or ST-GCN
artifact was modified.

## EXP029 — Observed Local Semantic BEV Sufficiency Audit

2026-08-31: completed the authorized Train/Val-only semantic BEV audit. A
real Stage-D s0/s1 smoke passed, followed by full BEV generation for 29,133
Train and 9,742 Val episodes. The representation used 77,750 unique visited
observations, exact s1-centered world projection, metric-depth ray occupancy,
and 15-channel `[15,80,80]` uint8 BEV features. Train-only cosine kNN reached
Val three-way/binary agreement of 0.426709/0.559844 at k=25; the compact probe
reached 0.477725 three-way and 0.631390 binary Val accuracy. Decision:
**INCONCLUSIVE** as a policy sufficiency result; no trajectory rollout was
performed. Only s0/s1 observations were rendered, p2/p3 and Test were never
accessed, and skeleton/RGB/perception/ST-GCN artifacts were untouched.

### EXP029-R1 correction

2026-08-31: replaced normalized depth rays with the exact non-normalized
Habitat `-Z` pinhole ray and validated center/left/right/top/bottom/corner
endpoints (maximum error 0.019115 m). Old Train/Val BEV caches were deleted
and regenerated with the same 4-worker, s0/s1-only protocol: 29,133 Train and
9,742 Val episodes. Corrected k=25 agreement was 0.430404 three-way / 0.559639
binary; local consistency mean/median were 0.499893/0.48, with 5.52% of
neighborhoods at least 0.8-consistent. Fixed-margin audits showed k=25
three-way agreement 0.481699/0.499669/0.515572/0.548125 at margins
≥0.25/0.5/1.0/2.0. Relative to EXP028's frozen 0.444570 k=25 agreement,
EXP029-R1 is **CASE B**: coarse observed semantic BEV does not resolve the
representation insufficiency. Test remained locked.

## EXP032--EXP034 — Overnight quality decomposition campaign

2026-09-01: completed the frozen Train/Val-only overnight audit on 29,133
Train and 9,742 Val Stage-D episodes. EXP032-A pose error decomposition is
blocked because the repository exposes no canonical GT-to-H36M17 mapping;
EXP032-B reused only frozen Stage-B/ST-GCN diagnostics and is therefore a
privileged, utility-definition-level sanity bound. Future CE correlated with
candidate utility at Pearson -0.728750 / Spearman -0.609646, while correctness
correlated at 0.477016 / 0.492565. EXP033 legal-input predictors reached Val
candidate CE Pearson/Spearman 0.465552/0.434138 (winner accuracy 0.566679)
and direct-utility 0.411064/0.277805 (winner accuracy 0.550057), showing weak
predictability from current state plus geometry alone. EXP034 Ridge R² was
0.121033 for the legal base and 0.555201 after adding privileged future
recognition quality (delta R² 0.434168); only ten rounded duplicate contexts
were available and neither three-way nor binary switch rates were nonzero.
This is CASE B as a definition-level recognition-quality signal, not causal
evidence; pose branches and Test remain locked, with no perception or Habitat
rerun.

## EXP035--EXP037 — Dense field and graph active-perception campaign

2026-09-01: completed the authorized frozen Train/Val-only campaign. EXP035
generated 589 Train and 197 Val 32-view fields with Stage-B smoke PASS (device
reproduction tolerance 1e-2). Val CE field mean/std were 4.906191/5.746398;
angular-neighbor and radial-neighbor correlations were only 0.283706 and
0.441162 (Pearson), indicating weak local smoothness. On 5,234 Val
Oracle-Move p2/p3 episodes, EXP036 winner accuracies were Dense 0.520443,
Bradley--Terry 0.503439, GMRF 0.529041, Bayesian mean/LCB 0.536301 and
Thompson mean 0.537639; the 32-view dense selection Top-1 hit was 0.040609.
EXP037 graph rollouts were completed for H=1/2/3 with Euclidean diagnostic
movement costs; the best non-privileged terminal CE was StaticDP H=3 at
1.637608, while the privileged Oracle H=3 terminal CE was 0.239832. Dense
supervision, pairwise ranking and fixed graph/Bayesian priors therefore did
not resolve the representation bottleneck (CASE E). All methods are offline
diagnostics, Test remained locked, and no perception/Habitat/ST-GCN training
or regeneration was performed.

## EXP035--EXP037-R1 — Context identity repair and HAR evaluation

2026-09-01: invalidated the prior record-id-only campaign and reran with the
canonical `(scene_id, region, record_id)` identity. The audit found 589/197
motion IDs expanding to 29,133/9,742 Train/Val context keys; Train/Val episode
and context overlaps were zero. All 217,475 Train and 72,784 Val Stage-B
candidate values passed full reproduction with maximum absolute log-probability
error 0.007377 (<1e-2), and the frozen EXP014 unified evaluator gate reproduced
Accuracy 0.6582540931 and Macro-F1 0.6101526052. EXP036-R1 final HAR Accuracy
for Dense/Bradley--Terry/GMRF/BayesianMean/BayesianLCB/Thompson was
0.645099/0.559091/0.647315/0.648388/0.648245/0.648388; corresponding mean
regrets were 1.478509/2.568556/1.473572/1.460368/1.462354/1.461173.
EXP037-R1 included all 13,987 episodes in terminal HAR evaluation; at H=3,
the best non-privileged Thompson method reached Accuracy 0.591335 (Macro-F1
0.520750), while frozen Stay reproduced Stage-C-v0. The old EXP035--EXP037
CASE E is therefore invalid due to identity collapse; R1 does not establish a
ranking or downstream HAR improvement over EXP014. Test remained locked.

## EXP038--EXP040 — Oracle observability and belief-space active HAR

2026-09-02: completed the authorized Train/Val-only campaign using canonical
dense recognition-quality fields and frozen ST-GCN observations. EXP038 legal
L0 reached Accuracy 0.656467 / Macro-F1 0.609065 / mean regret 1.419867; the
GT-label privileged L1 upper bound reached 0.702581 / 0.643642 / 1.139196,
while canonical GT motion mapping was unavailable and recorded as
GT_MOTION_STATE_BLOCKED. EXP039 current-belief Top-3 expected-risk was the
best legal view-risk method (Accuracy 0.647959, mean regret 1.484942), below
the frozen Stage-C-v0 trajectory; the GT-label head remained privileged.
EXP040 belief-space graph planning did not improve the legal baseline: the best
legal fused HAR was correctness-greedy H=2 at Accuracy 0.661114, whereas the
true-CE graph oracle H=3 reached terminal Accuracy 0.827197. Candidate heads
used the initial legal Stage-D state and beliefs were updated only after
visited transitions; this rapid-pilot approximation is recorded in result.json.
Test was not read and no perception/Habitat/ST-GCN training or regeneration
was performed.

## EXP041--EXP044 CUDA execution follow-up

2026-09-02: executed the fixed EXP042 A/B/C Train/Val campaign in the external
conda `habitat` environment on an RTX 4090 (CUDA 12.4). A/B/C used 29,133
Train contexts and 9,742 Val contexts; Val p2/p3 reconstruction MAE was
0.120761/0.121299/0.114823, with final Train losses 0.025537/0.025405/0.022301.
EXP043 frozen-ST-GCN selectors gave Accuracy 0.683921 (privileged label
diagnostic), 0.637878 (entropy), 0.638307 (top-1 confidence), and 0.643669
(belief cross-entropy). EXP044 was not executed: the current code has no
validated recurrent multi-step trajectory evaluator/history-rollout entry
point and records explicit `SKIPPED_REQUIRES_RECURRENT_WORLD_MODEL_ROLLOUT`
rather than fabricating metrics. Test remained locked; no Habitat/perception/
ST-GCN retraining was performed.

2026-09-03: completed EXP049/EXP050 counterfactual view-revision analysis in
the external habitat CUDA environment (RTX 4090), using frozen EXP046 cache
and no Test access. Legal candidate scaling was non-monotonic; JOINT_REVISION
at ALL_LEGAL reached Accuracy 0.686566 / Macro-F1 0.646064 versus independent
CF_CORRECTNESS_MLP 0.676628 / 0.629705, with paired ΔAccuracy 0.009938 (95%
CI [0.005148, 0.014728]) and ΔMacro-F1 0.016359 (95% CI [0.009153, 0.023854]).
The EXP051 authorization gate passed, but H=2 was blocked because frozen WM-E
requires RGB history for newly visited views and the approved cache lacks those
embeddings; no substitute inputs were used. Test, perception and upstream
model retraining remained disabled.
