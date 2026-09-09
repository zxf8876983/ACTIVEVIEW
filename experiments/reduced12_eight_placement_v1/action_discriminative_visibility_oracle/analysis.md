# Reduced12 action-discriminative visibility oracle

Val-only privileged diagnostic. Train is used only for frozen ST-GCN body-part masking importance; GT action and future visibility are not deployable inputs.

## Moving Val metrics

| Method | Accuracy | Macro-F1 | ΔAcc vs Frozen (pp) | ΔF1 (pp) |
| --- | ---: | ---: | ---: | ---: |
| S0-only | 0.254266 | 0.235500 | -20.000 | -20.928 |
| FrozenStageCv0 | 0.454266 | 0.444782 | +0.000 | +0.000 |
| SceneVisibility | 0.469544 | 0.456516 | +1.528 | +1.173 |
| HumanSelfVisibility | 0.262302 | 0.256464 | -19.196 | -18.832 |
| TotalVisibility | 0.364187 | 0.355554 | -9.008 | -8.923 |
| GTAction-SceneDiscriminative | 0.469246 | 0.456471 | +1.498 | +1.169 |
| GTAction-HumanDiscriminative | 0.263591 | 0.257107 | -19.067 | -18.767 |
| GTAction-TotalDiscriminative | 0.363790 | 0.353675 | -9.048 | -9.111 |
| AnyCorrect Oracle | 0.728274 | 0.729637 | +27.401 | +28.486 |

## Train masking importance top-3

| Action | Top-3 body parts |
| --- | --- |
| walk | torso (0.505), right_upper_arm (0.146), left_upper_arm (0.120) |
| sit | torso (0.699), right_thigh (0.111), right_upper_arm (0.084) |
| stand up | torso (0.555), right_upper_arm (0.179), right_forearm (0.084) |
| bend | torso (0.521), right_lower_leg (0.101), left_thigh (0.098) |
| crawl | right_thigh (0.707), torso (0.106), left_lower_leg (0.082) |
| stumble | torso (0.453), right_thigh (0.180), right_upper_arm (0.093) |
| clap | torso (0.750), right_forearm (0.133), right_upper_arm (0.060) |
| throw | torso (0.414), right_forearm (0.202), right_upper_arm (0.165) |
| kick | torso (0.652), right_thigh (0.143), right_upper_arm (0.082) |
| knock | torso (0.611), left_lower_leg (0.127), right_upper_arm (0.117) |
| punch | torso (0.382), right_thigh (0.314), right_upper_arm (0.100) |
| touching face | right_lower_leg (0.375), left_thigh (0.325), left_lower_leg (0.190) |

## Candidate score diagnostics

| Score | Spearman with GT true-logp | Correct mean | Wrong mean |
| --- | ---: | ---: | ---: |
| scene_visibility | 0.265521 | 0.877567 | 0.606815 |
| human_self_visibility | -0.057037 | 0.176153 | 0.187638 |
| total_visibility | 0.196420 | 0.150536 | 0.111785 |
| gt_action_scene_discriminative | 0.262011 | 0.895174 | 0.615000 |
| gt_action_human_discriminative | -0.065580 | 0.083609 | 0.093836 |
| gt_action_total_discriminative | 0.164687 | 0.072007 | 0.056140 |

## Scientific conclusion

The action-conditioned visibility scores are privileged upper-bound diagnostics, not deployable policies. In this run GTAction-TotalDiscriminative reaches 0.363790 accuracy, well below the 47–50% visibility-only range and the 55% target. The action-specific weights therefore do not rescue the human-visibility signal; visibility-only NBV is insufficient to explain the AnyCorrect gap. The next direction should predict candidate future recognizer evidence or utility rather than add more visibility heuristics.

Protocol flags: test_used=false; training_new_model_used=false; train_used_only_for_frozen_stgcn_masking_importance=true; gt_action_used_for_val_oracle_only=true; gt_future_visibility_used_for_oracle_only=true; deployable=false.
