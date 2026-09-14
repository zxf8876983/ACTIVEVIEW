# Yaw8Fair Strict-Frame0 NBV Full Re-baseline

Formal task: Frame0 → exactly one Stage-A legal candidate → selected real O1 alone → Yaw8Fair HAR.
Formal recognizer: frozen Yaw8 ST-GCN encoder + frozen matched Policy-balanced shared head.
Policy Test: locked; Moving Val is evaluation-only; no new perception data were generated.

## Main Moving-Val results

| Method | Accuracy | Macro-F1 | Gain vs Random | Gain vs StaticPrior |
|---|---:|---:|---:|---:|
| Stay | 0.350496 | 0.364253 | -7.629pp | -19.940pp |
| Random legal | 0.426786 | 0.450423 | +0.000pp | -12.312pp |
| StaticViewPrior | 0.549901 | 0.571990 | +12.312pp | +0.000pp |
| Frame0SceneVisibility | 0.583929 | 0.598955 | +15.714pp | +3.403pp |
| GeometryOnly-Visibility | 0.540377 | 0.561212 | +11.359pp | -0.952pp |
| RGBGlobal-Visibility | 0.567361 | 0.582567 | +14.058pp | +1.746pp |
| RGBSpatial-Visibility | 0.555456 | 0.571285 | +12.867pp | +0.556pp |
| GeometryOnly-TrueLogP | 0.531448 | 0.555499 | +10.466pp | -1.845pp |
| RGBGlobal-TrueLogP | 0.545933 | 0.565164 | +11.915pp | -0.397pp |
| RGBGlobal-Margin | 0.549504 | 0.568928 | +12.272pp | -0.040pp |
| RGBGlobal-TrueLogP+VisibilityAux | 0.548512 | 0.566333 | +12.173pp | -0.139pp |
| Prior+GeometryResidual | 0.541270 | 0.564201 | +11.448pp | -0.863pp |
| Prior+RGBResidual λ=0.5 | 0.550794 | 0.573851 | +12.401pp | +0.089pp |
| Prior+RGBResidual λ=1.0 | 0.547520 | 0.569810 | +12.073pp | -0.238pp |
| GT-TrueLogP Oracle | 0.760714 | 0.775961 | +33.393pp | +21.081pp |
| GT-Margin Oracle | 0.776190 | 0.796334 | +34.940pp | +22.629pp |

## Re-baseline decision

Best strict method: **Frame0SceneVisibility**, Accuracy/F1=0.583929/0.598955; gain over StaticPrior=+3.403pp.
Best learned branch: **Prior+RGBResidual λ=0.5**, Accuracy/F1=0.550794/0.573851; gain over StaticPrior=+0.089pp.
Pre-registered decision: **KEEP INSTANCE-CONDITIONED FRAME0 NBV**.
Candidate-only oracle sanity: GT-TrueLogP=0.760714, GT-Margin=0.776190, AnyCorrect=0.776190.

## Causal and training protocol

Policy Train contexts=46324; Moving Val contexts=10080; mean legal candidates=6.816.
All selector branches use a deterministic 10% Policy-Train record holdout for checkpoint selection; Moving Val was not used for training or selection.
Candidate observations/features/logits and GT labels are never selector inputs; they are target/oracle/terminal-only.

## RGB / geometry shuffle and occlusion

Best learned RGB shuffle drop=+1.667pp; geometry shuffle drop=+0.724pp.
High-occlusion subset size=3271; best learned gain over StaticPrior=+0.336pp.

## Scientific conclusion

Historical old-recognizer values are reference-only; the formal comparison is entirely within Yaw8Fair.
Conclusion: **KEEP INSTANCE-CONDITIONED FRAME0 NBV**. No automatic follow-up method was started.

```text
policy_test_used=false
moving_val_used_for_training_or_selection=false
candidate_observation_used_for_selector=false
gt_action_used_for_selector=false
new_rgb_generated=false
new_skeleton_generated=false
yaw8_encoder_modified=false
yaw8_shared_head_modified=false
```
