# Recognizer-Level Clean Motion Reference Audit

Val Moving contexts only. Clean references use exact raw-val AMASS intervals, MotionConverter and Habitat articulated-humanoid FK; no candidate estimated skeleton was used to construct a clean reference.

| Method | Accuracy | Macro-F1 | Move rate |
|---|---:|---:|---:|
| S0-only | 0.254266 | 0.235500 | 0.000000 |
| FrozenStageCv0 | 0.454266 | 0.444782 | 1.000000 |
| Candidate-Conditioned Spatial | 0.454266 | 0.444782 | 1.000000 |
| Real-GTMargin Oracle | 0.709623 | 0.704598 | 1.000000 |
| AnyCorrect Oracle | 0.728274 | 0.729637 | 0.000000 |
| MaxFeatureCos-to-Clean | 0.352679 | 0.332132 | 1.000000 |
| MinJSD-to-Clean | 0.367460 | 0.334927 | 1.000000 |

## Clean recognizer

Unique-record clean ST-GCN: Accuracy=0.266667, Macro-F1=0.185584 (N=105).
Moving-context weighted clean ST-GCN: Accuracy=0.245238, Macro-F1=0.178535 (N=10080).

## Representation and ranking

- FeatureCos: global Pearson=0.117758, Spearman=0.104015; within-context Spearman mean=0.102751.
- NegativeFeatureL2: global Pearson=-0.140761, Spearman=-0.086012; within-context Spearman mean=-0.109227.
- NegativePosteriorJSD: global Pearson=-0.266490, Spearman=-0.224632; within-context Spearman mean=-0.201869.
- SceneVisibility: global Pearson=0.219863, Spearman=0.276370; within-context Spearman mean=0.217956.
- ProjectedArea: global Pearson=0.121937, Spearman=0.188695; within-context Spearman mean=0.129703.
- PoseConfidence: global Pearson=0.214855, Spearman=0.272156; within-context Spearman mean=0.220586.
- NegativeDistance: global Pearson=-0.145692, Spearman=-0.184023; within-context Spearman mean=-0.146220.
- NegativeMotionDeviation: global Pearson=-0.030958, Spearman=-0.134797; within-context Spearman mean=-0.133424.
- Feature cosine mean=0.714210; normalized feature L2 mean=0.714295; posterior JSD mean=0.537188.

## Selected-vs-GT-best audit

Clean-correct rate=0.245238; GT-margin oracle-correct rate=0.709623.
CleanWrong→OracleCorrect contexts=5149; CleanCorrect→OracleWrong contexts=468.
Frozen selection wrong while oracle correct=2574; severe-miss contexts=1287.

## Scientific interpretation

The clean recognizer is a representation ceiling reference: its score is reported independently of ActiveView selector decisions. The estimated candidate and clean distributions are not interchangeable, so the remaining gap should be interpreted as both motion/observation mismatch and candidate selection error.
Candidate-conditioned Spatial is represented by the archived Stage-C proposal_rank_1_id; GT-GTMargin Oracle selects the legal candidate with maximum true-class margin.

Flags: `test_used=false`; `training_used=false`; `new_rgb_rendered=false`; `new_pose_estimation=false`; `clean_h36m17_source=exact AMASS->Habitat FK`; `clean_reference_used_for_privileged_reference_only=true`; `deployable=false`.
