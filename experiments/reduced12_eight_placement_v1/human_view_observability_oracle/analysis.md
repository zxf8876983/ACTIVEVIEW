# Reduced12 Human View Observability Oracle

Val Moving contexts only. GT world-space H36M17 joints were projected into legal candidate cameras using the fixed 256x256/HFOV 75 degree protocol. This is a privileged geometry diagnostic; future RGB and estimated future skeletons were not used for selection.

| Method | Accuracy | Macro-F1 | ΔAcc vs Frozen (pp) | ΔF1 vs Frozen (pp) | Move rate |
|---|---:|---:|---:|---:|---:|
| S0-only | 0.254266 | 0.235500 | -20.000 | -20.928 | 0.000000 |
| FrozenStageCv0 | 0.454266 | 0.444782 | +0.000 | +0.000 | 1.000000 |
| SceneVisibility | 0.469544 | 0.456516 | +1.528 | +1.173 | 1.000000 |
| MaxProjectedArea | 0.425496 | 0.413591 | -2.877 | -3.119 | 1.000000 |
| MaxLimbProjection | 0.433036 | 0.426513 | -2.123 | -1.827 | 1.000000 |
| MaxJointSeparation | 0.318948 | 0.302122 | -13.532 | -14.266 | 1.000000 |
| HumanObservability | 0.424206 | 0.413982 | -3.006 | -3.080 | 1.000000 |
| SceneHumanObservability | 0.455556 | 0.442970 | +0.129 | -0.181 | 1.000000 |
| AnyCorrect Oracle | 0.728274 | 0.729637 | +27.401 | +28.486 | 0.474008 |

## Candidate diagnostics

- **projected_area**: Spearman with GT-class true logp `0.188694`; correct-candidate mean `0.216695` vs wrong `0.168141`; indistinguishable-context rate `0.000000`.
- **limb_projection**: Spearman with GT-class true logp `0.152540`; correct-candidate mean `77.146963` vs wrong `69.962226`; indistinguishable-context rate `0.000000`.
- **joint_separation**: Spearman with GT-class true logp `0.114668`; correct-candidate mean `0.288578` vs wrong `0.273570`; indistinguishable-context rate `0.000000`.
- **human_observability**: Spearman with GT-class true logp `0.125916`; correct-candidate mean `1.519010` vs wrong `1.271879`; indistinguishable-context rate `0.000000`.
- **scene_visibility**: Spearman with GT-class true logp `0.265521`; correct-candidate mean `0.877567` vs wrong `0.606815`; indistinguishable-context rate `0.188294`.
- **scene_human_observability**: Spearman with GT-class true logp `0.233161`; correct-candidate mean `0.611814` vs wrong `0.451185`; indistinguishable-context rate `0.007242`.

## Scientific judgment

HumanObservability does not materially exceed FrozenStageCv0 (-3.006 pp Accuracy); this privileged geometry-only score is not sufficient as a selector.
The fixed Scene+Human combination gives no material additional gain over Frozen (+0.129 pp Accuracy); no weight tuning was performed.
For reference, SceneVisibility alone changes Accuracy by +1.528 pp versus Frozen; if all geometry-only gains remain small, the next diagnostic should examine realistic human self-occlusion/view-dependent observability rather than tune weights.

Flags: `test_used=false`, `training_used=false`, `gt_future_human_geometry_used_for_oracle_only=true`, `future_candidate_rgb_used=false`, `future_candidate_skeleton_used_only_for_terminal_evaluation=true`, `deployable=false`.
