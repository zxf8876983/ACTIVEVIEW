# Real / Predicted Recognition Evidence Decision Decomposition

Train/Val only; policy Test was not read. Existing archived skeleton, true candidate evidence, visited s0 DINO and Feature+LogP predicted-evidence assets were reused; no RGB/skeleton/DINO regeneration was performed.

## Moving Val

| Method | Accuracy | Macro-F1 | ΔAcc vs Frozen | ΔF1 vs Frozen |
|---|---:|---:|---:|---:|
| S0-only | 0.254266 | 0.235500 | -20.000pp | -20.928pp |
| FrozenStageCv0 | 0.454266 | 0.444782 | +0.000pp | +0.000pp |
| Candidate-Conditioned Spatial | 0.471528 | 0.463723 | +1.726pp | +1.894pp |
| SceneVisibility | 0.469544 | 0.456516 | +1.528pp | +1.173pp |
| AnyCorrect Oracle | 0.728274 | 0.729637 | +27.401pp | +28.486pp |
| Real-MaxProb | 0.465278 | 0.448771 | +1.101pp | +0.399pp |
| Real-Entropy | 0.197520 | 0.194679 | -25.675pp | -25.010pp |
| Real-Margin | 0.465873 | 0.449905 | +1.161pp | +0.512pp |
| Real-S0Agreement | 0.304762 | 0.291921 | -14.950pp | -15.286pp |
| Real-MeanFusion | 0.229365 | 0.233892 | -22.490pp | -21.089pp |
| Real-ProductFusion | 0.231647 | 0.244531 | -22.262pp | -20.025pp |
| Real-GTTrueLogP | 0.709325 | 0.704330 | +25.506pp | +25.955pp |
| Real-GTMargin | 0.709623 | 0.704598 | +25.536pp | +25.982pp |
| real_correctness_bce | 0.498611 | 0.480887 | +4.435pp | +3.611pp |
| real_gt_margin_listwise | 0.494147 | 0.490256 | +3.988pp | +4.547pp |
| predicted_correctness_bce | 0.398016 | 0.383990 | -5.625pp | -6.079pp |
| predicted_gt_margin_listwise | 0.404960 | 0.395532 | -4.931pp | -4.925pp |

## Full Val references

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| S0-only | 0.329601 | 0.329175 |
| FrozenStageCv0 | 0.460167 | 0.450040 |
| AnyCorrect Oracle | 0.731338 | 0.731446 |

## Evidence/ranker diagnostics

- **Real-MaxProb** candidate-score Spearman vs GT true-logp: 0.253058.
- **Real-Entropy** candidate-score Spearman vs GT true-logp: -0.246343.
- **Real-Margin** candidate-score Spearman vs GT true-logp: 0.261004.
- **Real-S0Agreement** candidate-score Spearman vs GT true-logp: 0.147959.
- **Real-MeanFusion** candidate-score Spearman vs GT true-logp: -0.167164.
- **Real-ProductFusion** candidate-score Spearman vs GT true-logp: -0.257563.
- **Real-GTTrueLogP** candidate-score Spearman vs GT true-logp: 1.000000.
- **Real-GTMargin** candidate-score Spearman vs GT true-logp: 0.998979.
- **RealEvidence-CorrectnessBCE** AUROC/AP=0.799208/0.704180; context ranking Spearman=0.265987; selected correctness=0.498611.
- **RealEvidence-GTMarginListwise** AUROC/AP=0.666982/0.545708; context ranking Spearman=0.350082; selected correctness=0.494147.
- **PredictedEvidence-CorrectnessBCE** AUROC/AP=0.790080/0.641703; context ranking Spearman=0.106592; selected correctness=0.398016.
- **PredictedEvidence-GTMarginListwise** AUROC/AP=0.532738/0.307071; context ranking Spearman=0.363852; selected correctness=0.404960.

## Gap decomposition

- Real GTTrueLogP minus best real non-GT selector: +24.345pp Accuracy; +25.443pp Macro-F1.
- Best real learned minus best predicted learned: +9.365pp Accuracy; +9.472pp Macro-F1.
- AnyCorrect Oracle minus best predicted selector: +32.331pp Accuracy; +33.410pp Macro-F1.

## Scientific judgment

Real future recognition evidence contains substantial recoverable information, but non-GT utility selection remains the dominant decision bottleneck.

No formal WM/JR/ST-GCN checkpoint, taxonomy, or split was changed. No Test data were read.

`test_used=false`; `train_used_for_ranker_training=true`; `future_candidate_skeleton_used_for_target_and_terminal_eval_only=true`; `future_candidate_rgb_used=false`; `future_candidate_dino_used=false`.
