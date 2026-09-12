# Frame-0 Alternative-View Visibility Predictor

- Training split: Policy Train; evaluation/model-selection split: Moving Val.
- Policy Test used: false.
- Target: frame-0 scene-only H36M17 visibility ratio from Habitat raycasts.
- Predictor inputs: current frame-0 RGB DINO representation and Stage-A candidate geometry only.
- Future motion/candidate observation/action label/recognizer output used by predictor: false.
- Terminal recognizer: frozen pretrained ST-GCN plus frozen shared adapted head.

## Moving-Val results

| Selector | Accuracy | Macro-F1 | Move rate |
|---|---:|---:|---:|
| Stay | 0.302579 | 0.292976 | 0.000000 |
| Random legal | 0.365079 | 0.364567 | 0.843254 |
| Frame0SceneVisibility Oracle | 0.489782 | 0.485962 | 0.523611 |
| GT-TrueLogP Oracle | 0.753175 | 0.755358 | 0.886409 |
| Historical Route-1 + Shared | 0.509524 | 0.509806 | 0.972024 |
| GeometryOnly | 0.477579 | 0.479837 | 0.873413 |
| RGBGlobal+Geometry | 0.494841 | 0.493881 | 0.653671 |
| RGBSpatial+Geometry | 0.493849 | 0.494691 | 0.806845 |

## Visibility prediction and ranking

| Branch | MAE | RMSE | Candidate Spearman | Within-context Spearman | Oracle top-1 overlap | Stay/move agreement |
|---|---:|---:|---:|---:|---:|---:|
| GeometryOnly | 0.351519 | 0.400998 | 0.384283 | 0.165653 | 0.374206 | 0.567857 |
| RGBGlobal+Geometry | 0.224332 | 0.283694 | 0.691010 | 0.362247 | 0.660813 | 0.819544 |
| RGBSpatial+Geometry | 0.191582 | 0.261669 | 0.714893 | 0.434291 | 0.512996 | 0.691766 |

High-occlusion subset: bottom tertile of Stay target (3271 contexts).

| Selector | High-occlusion Acc | High-occlusion Macro-F1 |
|---|---:|---:|
| Stay | 0.102109 | 0.050838 |
| Random legal | 0.269337 | 0.268728 |
| GeometryOnly | 0.408744 | 0.421466 |
| RGBGlobal+Geometry | 0.464384 | 0.469584 |
| RGBSpatial+Geometry | 0.453684 | 0.458357 |
| Frame0SceneVisibility Oracle | 0.468053 | 0.470394 |
| Historical Route-1 + Shared | 0.448792 | 0.455826 |

Best deployable predictor: **RGBGlobal+Geometry**, Accuracy=0.494841, GainRecovery=104.1% (Random=0.365079; frame-0 oracle=0.489782).
Decision: **STRONG KEEP** under the preregistered Acc<0.43 or GainRecovery<50% kill rule, Acc>=0.45/GainRecovery>=65% keep rule, and Acc>=0.465 strong-keep rule.

## Interpretation

GeometryOnly and RGB branches are compared under identical target/action-set/training budgets. RGBGlobal is mean-pooled from strict frame-0 DINO spatial tokens because the frozen cache stores 4x4 tokens; RGBSpatial preserves token structure through a small frozen-token projection.
The RGB branches improve visibility ranking over GeometryOnly; RGBGlobal is the best downstream selector by Accuracy, while RGBSpatial has the lowest visibility MAE and highest candidate-level/within-context rank correlation. Error-transition details are in `context_transitions.json`.

```text
policy_test_used=false
training_split=Policy Train
evaluation_split=Moving Val
future_candidate_rgb_used=false
future_candidate_skeleton_used_for_predictor=false
gt_action_used=false
recognizer_output_used_for_predictor=false
```
