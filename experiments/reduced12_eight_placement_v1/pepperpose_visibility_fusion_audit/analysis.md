# PepperPose confidence + visibility fusion audit

This is a privileged, non-deployable diagnostic. The true-facing frame-0 confidence table is loaded from the preceding audit; no angle sanity or confidence landscape was recomputed.

## Moving Val results

| Method | Accuracy | Macro-F1 | Move rate |
|---|---:|---:|---:|
| RGBGlobal-Visibility | 0.567361 | 0.582567 | 1.000000 |
| RGBGlobal-Visibility+Confidence | 0.567361 | 0.583573 | 1.000000 |
| Frame0SceneVisibility | 0.583929 | 0.598955 | 1.000000 |
| Frame0SceneVisibility+Confidence | 0.502579 | 0.524444 | 1.000000 |
| GTYaw-PoseConfidencePrior | 0.437996 | 0.465250 | 1.000000 |
| GT-TrueLogP Oracle | 0.760714 | 0.775961 | 1.000000 |

RGBGlobal selected λ=0.25; Moving gain=+0.000 pp Accuracy.
Frame0SceneVisibility selected λ=0.25; Moving gain=-8.135 pp Accuracy.

## Complementarity

- **RGBGlobal_vs_Confidence**: selected-view agreement 0.315873; primary-correct/confidence-wrong 1970; primary-wrong/confidence-correct 666; both wrong 3695; pair AnyCorrect 6385 (0.633433).
- **Frame0SceneVisibility_vs_Confidence**: selected-view agreement 0.325000; primary-correct/confidence-wrong 2010; primary-wrong/confidence-correct 539; both wrong 3655; pair AnyCorrect 6425 (0.637401).

## Disagreement subsets

- **RGBGlobal_vs_Confidence** (6896 contexts): primary Acc=0.567285, confidence Acc=0.378190, fused Acc=0.567285.
- **Frame0SceneVisibility_vs_Confidence** (6804 contexts): primary Acc=0.584803, confidence Acc=0.368607, fused Acc=0.506467.

## High-occlusion subset

Strict lower tertile of current frame-0 Stay SceneVisibility: n=3271.

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| RGBGlobal | 0.515439 | 0.534703 |
| RGBGlobal+Confidence | 0.514827 | 0.535387 |
| Frame0SceneVisibility | 0.520024 | 0.539481 |
| Frame0SceneVisibility+Confidence | 0.485173 | 0.507034 |
| GTYaw-PoseConfidencePrior | 0.346989 | 0.372445 |
| GT-TrueLogP Oracle | 0.664934 | 0.689528 |

## Decision: KILL ENTIRE PEPPERPOSE BRANCH

The confidence prior is evaluated with GT body yaw and is therefore not deployable. Lambda selection used only a deterministic Policy-Train record holdout; Moving Val was not used for tuning. The policy archive confidence limitation from the previous audit remains: it stores sequence-level `(32,)` confidence, while the frozen prior table itself is frame-0-derived from the internal Yaw8 archive.

Flags: `policy_test_used=false`, `training_used=false`, `new_rgb_generated=false`, `future_candidate_observation_used_for_selection=false`, `deployable=false`.
