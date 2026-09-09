# Reduced12 3D Scene Joint Visibility Oracle

Val Moving contexts only. Static HM3D geometry was ray-cast from each candidate camera to six reconstructed GT world-space H36M17 poses. The ray-cast simulator contains no humanoid, so this is environmental occlusion only; the future archived skeleton is used only for terminal recognizer evaluation.

| Method | Accuracy | Macro-F1 | ΔAcc vs Frozen (pp) | ΔF1 vs Frozen (pp) | Move rate |
|---|---:|---:|---:|---:|---:|
| S0-only | 0.254266 | 0.235500 | -20.000 | -20.928 | 0.000000 |
| FrozenStageCv0 | 0.454266 | 0.444782 | +0.000 | +0.000 | 1.000000 |
| TopDownLOS-Filter+Frozen | 0.468948 | 0.459178 | +1.468 | +1.440 | 1.000000 |
| SceneVisibility | 0.469544 | 0.456516 | +1.528 | +1.173 | 1.000000 |
| SceneVisibility-Filter+Frozen | 0.468155 | 0.457225 | +1.389 | +1.244 | 1.000000 |
| AnyCorrect Oracle | 0.728274 | 0.729637 | +27.401 | +28.486 | 0.474008 |

## Visibility diagnostics

Correct candidate mean visibility: 0.877567; wrong candidate mean: 0.606815.
Candidate Spearman(scene visibility, GT-class true log-probability): 0.265521.
Mean within-context visibility range: 0.564522; indistinguishable contexts: 0.188294.
Top-down LOS=1 candidate fraction: 0.504381; all-LOS-equal contexts: 0.315575.

## Scientific judgment

SceneVisibility gives a material gain over FrozenStageCv0 (+1.528 pp Accuracy), indicating that true 3D environmental joint visibility has NBV value in this diagnostic.
Median visibility filtering plus Frozen ranking improves the baseline (+1.389 pp Accuracy), suggesting complementary geometry and HAR ranking signals.
If gains remain small, the next diagnostic should examine view-dependent human observability and self-occlusion rather than tune this visibility score.

Flags: `test_used=false`, `training_used=false`, `habitat_gt_scene_geometry_used=true`, `gt_future_human_world_joints_used_for_oracle_only=true`, `future_candidate_rgb_used=false`, `future_candidate_skeleton_used_only_for_terminal_evaluation=true`, `deployable=false`.
