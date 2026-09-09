# Reduced12 Top-down LOS Filter + FrozenStageCv0

Val Moving only. The binary Habitat top-down LOS is a privileged geometry rejector; FrozenStageCv0's existing Stage-C predicted utility remains the only fine-ranking score. No model was trained and no Test/perception artifact was read or generated.

| Method | Accuracy | Macro-F1 | ΔAcc vs Frozen (pp) | ΔF1 vs Frozen (pp) | Move rate |
|---|---:|---:|---:|---:|---:|
| S0-only | 0.254266 | 0.235500 | -20.000 | -20.928 | 0.000000 |
| FrozenStageCv0 | 0.454266 | 0.444782 | +0.000 | +0.000 | 1.000000 |
| TopDown-LOS | 0.425397 | 0.413444 | -2.887 | -3.134 | 1.000000 |
| TopDownLOS-Filter+Frozen | 0.468849 | 0.459385 | +1.458 | +1.460 | 1.000000 |
| AnyCorrect Oracle | 0.728274 | 0.729637 | +27.401 | +28.486 | 0.474008 |

Contexts with at least one LOS=1 candidate: 0.820238; Frozen selections filtered out: 0.101389.
Among filtered contexts, rescue=272, harm=125, net=147.
Frozen selected correctness: P(correct|LOS=1)=0.512973; P(correct|LOS=0)=0.304164.

## Scientific judgment

The LOS filter materially improves FrozenStageCv0 (+1.458 pp Accuracy, +1.460 pp Macro-F1), supporting a coarse-geometry rejector plus HAR fine-ranking route.
LOS and Frozen ranking show some complementarity at the candidate level, but the end-to-end combination determines whether that signal is useful.

`test_used=false`, `training_used=false`, `habitat_gt_geometry_used_for_oracle_diagnostic=true`, and `future_candidate_skeleton_used_only_for_terminal_evaluation=true`.
