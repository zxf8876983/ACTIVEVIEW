# Reduced12 Top-down LOS NBV Oracle

Val Moving contexts only. This is a privileged geometry diagnostic: Habitat GT navmesh top-down slices rank archived legal candidates. It is not a joint-visibility measurement and no model, perception cache, RGB, skeleton, DINO or Test artifact was generated/read.

## Moving metrics

| Method | Accuracy | Macro-F1 | ΔAcc vs Frozen (pp) | ΔF1 vs Frozen (pp) | Move rate |
|---|---:|---:|---:|---:|---:|
| S0-only | 0.254266 | 0.235500 | -20.000 | -20.928 | 0.000000 |
| FrozenStageCv0 | 0.454266 | 0.444782 | +0.000 | +0.000 | 1.000000 |
| Random | 0.322619 | 0.316416 | -13.165 | -12.837 | 0.843254 |
| TopDown-LOS | 0.425397 | 0.413444 | -2.887 | -3.134 | 1.000000 |
| AnyCorrect Oracle | 0.728274 | 0.729637 | +27.401 | +28.486 | 0.474008 |

## LOS diagnostics

Mean legal-candidate LOS=1 fraction: 0.485743; median: 0.500000.
P(correct | LOS=1) = 0.458138; P(correct | LOS=0) = 0.208116; difference = +0.250021.
3181 / 0.315575 of contexts have identical LOS for all legal candidates.

## Interpretation
TopDown-LOS does not materially exceed FrozenStageCv0 (-2.887 pp Accuracy, -3.134 pp Macro-F1).
LOS separates correct from wrong archived candidates to a noticeable degree, although it remains a privileged cue.
Most contexts have at least some LOS variation, so the coarse signal is not entirely degenerate; its policy value is nevertheless limited by the measured gain.

All oracle quantities use only archived terminal recognition for evaluation. `test_used=false`, `training_used=false`, `future_candidate_rgb_used=false`, `future_candidate_skeleton_used_only_for_terminal_evaluation=true`, and `habitat_gt_geometry_used_for_oracle_diagnostic=true`.
