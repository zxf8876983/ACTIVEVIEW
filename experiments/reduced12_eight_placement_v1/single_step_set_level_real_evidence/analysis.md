# Single-Step Set-Level Real-Evidence Upper-Bound Study

Train/Val only. The common action set is stay + legal move candidates; policy Test was not read. No RGB, skeleton, DINO, ST-GCN, WM-E or JR artifact was regenerated or modified.

## Moving Val

| Method | Accuracy | Macro-F1 | ΔAcc vs Frozen | ΔF1 vs Frozen | Move rate |
|---|---:|---:|---:|---:|---:|
| S0-only | 0.254266 | 0.235500 | -20.000pp | -20.928pp | 0.000000 |
| FrozenStageCv0 | 0.454266 | 0.444782 | +0.000pp | +0.000pp | 1.000000 |
| Candidate-Conditioned Spatial | 0.471528 | 0.463723 | +1.726pp | +1.894pp | 0.922520 |
| ConsensusMean | 0.449306 | 0.432720 | -0.496pp | -1.206pp | 0.843452 |
| ConsensusMedian | 0.412401 | 0.394885 | -4.187pp | -4.990pp | 0.839980 |
| VoteConsensus | 0.405060 | 0.389518 | -4.921pp | -5.526pp | 0.837004 |
| ConfidenceWeightedConsensus | 0.473909 | 0.455187 | +1.964pp | +1.041pp | 0.852083 |
| AnyCorrect Oracle | 0.728274 | 0.729637 | +27.401pp | +28.486pp | 0.474008 |
| Real-GTTrueLogP Oracle | 0.727976 | 0.721495 | +27.371pp | +27.671pp | 0.886310 |
| Real-GTMargin Oracle | 0.728274 | 0.722059 | +27.401pp | +27.728pp | 0.887103 |
| RealEvidence-CorrectnessBCE | 0.501587 | 0.482541 | +4.732pp | +3.776pp | 0.907143 |
| RealEvidence-GTMarginListwise | 0.502778 | 0.499828 | +4.851pp | +5.505pp | 0.888294 |
| RealEvidence-SetAction | 0.448214 | 0.433923 | -0.605pp | -1.086pp | 0.846528 |
| RealEvidence-SetActionSoft | 0.465476 | 0.451511 | +1.121pp | +0.673pp | 0.839782 |
| RealEvidence-SetCorrectness | 0.494742 | 0.476583 | +4.048pp | +3.180pp | 0.839187 |

## Oracle alignment

AnyCorrect=0.728274; GTTrueLogP=0.727976; GTMargin=0.728274.

## Set diagnostics

SetActionClassifier Accuracy/Macro-F1: 0.452579/0.441161.
Best set selector: **RealEvidence-SetCorrectness**, 0.494742/0.476583.
Action-inference gap: +26.250pp; set-decision gap: +23.323pp; gain over aligned candidate-independent BCE: -0.685pp.

All set-level real-evidence selectors remain below 55%; even jointly observing true future evidence does not reliably recover the correct action in this single-step formulation. Further set-model scaling is not justified by this batch.

The aligned stay-inclusive oracle values are reported explicitly; any residual differences from older candidate-only oracle numbers are due to the corrected action set, not a protocol change.

`test_used=false`; `train_used_for_set_models=true`; `predicted_future_evidence_used=false`; `deployable=false`.
