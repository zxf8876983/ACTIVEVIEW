# Reduced12 all-cue learned fusion audit

Train/Val only. No policy Test was read; no RGB, DINO, skeleton, ST-GCN, WM-E or JR artifact was modified.

## Reference and deployable branches

| Method | Accuracy | Macro-F1 | ΔAcc vs Frozen | ΔAcc vs Candidate Spatial |
|---|---:|---:|---:|---:|
| S0-only | 0.254266 | 0.235500 | -20.000 pp | -21.726 pp |
| FrozenStageCv0 | 0.454266 | 0.444782 | +0.000 pp | -1.726 pp |
| Candidate-Conditioned Spatial | 0.471528 | 0.463723 | +1.726 pp | +0.000 pp |
| Top3-GTSequentialVerifier | 0.642163 | 0.634401 | +18.790 pp | +17.063 pp |
| AnyCorrect Oracle | 0.728274 | 0.729637 | +27.401 pp | +25.675 pp |
| SceneVisibility | 0.469544 | 0.456516 | +1.528 pp | -0.198 pp |
| Real-GTMargin Oracle | 0.709623 | 0.704598 | +25.536 pp | +23.810 pp |
| Real-GTTrueLogP | 0.709325 | 0.704330 | +25.506 pp | +23.780 pp |
| RealEvidence-GTMarginListwise | 0.502778 | 0.499828 | +4.851 pp | +3.125 pp |
| RealEvidence-CorrectnessBCE | 0.501587 | 0.482541 | +4.732 pp | +3.006 pp |
| G0_Base-Linear-Margin | 0.470139 | 0.460293 | +1.587 pp | -0.139 pp |
| G0_Base-Linear-Correctness | 0.470833 | 0.460626 | +1.657 pp | -0.069 pp |
| G0_Base-Mlp-Margin | 0.471032 | 0.460703 | +1.677 pp | -0.050 pp |
| G0_Base-Mlp-Listwise | 0.472123 | 0.466011 | +1.786 pp | +0.060 pp |
| G1_Base_CurrentDINO-Linear-Margin | 0.470536 | 0.461416 | +1.627 pp | -0.099 pp |
| G1_Base_CurrentDINO-Linear-Correctness | 0.472024 | 0.462185 | +1.776 pp | +0.050 pp |
| G1_Base_CurrentDINO-Mlp-Margin | 0.473512 | 0.464208 | +1.925 pp | +0.198 pp |
| G1_Base_CurrentDINO-Mlp-Listwise | 0.470933 | 0.465458 | +1.667 pp | -0.060 pp |
| G6_DeployableAll-Linear-Margin | 0.447222 | 0.433421 | -0.704 pp | -2.431 pp |
| G6_DeployableAll-Linear-Correctness | 0.461310 | 0.448949 | +0.704 pp | -1.022 pp |
| G6_DeployableAll-Mlp-Margin | 0.462401 | 0.453934 | +0.813 pp | -0.913 pp |
| G6_DeployableAll-Mlp-Listwise | 0.476290 | 0.469256 | +2.202 pp | +0.476 pp |

## Group ablation

G0 Base, G1 Base+current DINO, and G6 Deployable All were trained with complete Train/Val coverage. The coarse group ablation is reported with the pre-registered Linear-Margin and MLP-Listwise branches; the other model rows are retained as the requested deployable branch comparison.
G2–G5 privileged physical groups are NOT AVAILABLE: existing visibility/pose caches cover Moving Val only, so no Train leakage or expensive re-rendering was introduced.

Best completed branch: **G6_DeployableAll-Mlp-Listwise** (0.476290 Acc / 0.469256 Macro-F1).

## Scientific answers

Q1. SceneVisibility alone is 0.469544 Acc / 0.456516 Macro-F1; learned fusion is compared against it, but no additional SceneVisibility Train cue was trained because Train coverage is absent.
Q2. HumanSelfVisibility and other privileged physical groups are unavailable for a valid Train/Val fusion; no Val-only estimate is reported.
Q3. ProjectedArea/LimbProjection/JointSeparation are likewise unavailable with Train coverage, so this audit does not claim nonlinear gains from them.
Q4. Privileged All-Cue versus Deployable All-Cue cannot be numerically compared without Train physical-cue coverage; the privileged groups are explicitly marked NOT_AVAILABLE.
Q5. The best complete-coverage deployable branch remains below 0.50, so it does not approach the RealEvidence or oracle references.

All complete-coverage deployable fusion branches remain below 0.50; existing deployable cues do not yet establish a strong joint utility predictor.
Linear coefficients are correlation-only standardized weights, not causal attribution.

test_used=false; real_future_stgcn_evidence_as_input=false; deployable_branch_uses_future_gt_cues=false.
