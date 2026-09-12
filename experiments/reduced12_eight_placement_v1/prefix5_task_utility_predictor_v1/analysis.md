# Prefix-5 Task-Utility Selector

Train split: Policy Train (record-balanced, 313 records × 16 contexts/epoch)
Model-selection split: Moving Val
Policy Test used: false

Protocol: discrete-time view-switch approximation; prefix L=5 uses current skeleton frames 0:5.
Selector input: current 0:5 skeleton prefix, current frame-0 DINO global mean (branch C only), and Stage-A current/Stay + legal candidate geometry.
No future candidate observation, skeleton, RGB, DINO, confidence, recognizer output, hard action or GT label enters selector inference.
Target: strict mixed sequence current[0:5] + candidate[5:30]; Stay is current[0:30]. Terminal uses frozen ST-GCN encoder + frozen shared head.

## Moving Val results

| Method | Accuracy | Macro-F1 | Move rate |
|---|---:|---:|---:|
| Stay | 0.302579 | 0.292976 | 0.000000 |
| Random-ShortPrefix5 | 0.342262 | 0.329111 | 0.843254 |
| GeometryOnly-Mixed5 | 0.422817 | 0.406814 | 0.812103 |
| Prefix5+Geometry | 0.429266 | 0.411469 | 0.859524 |
| Prefix5+RGBGlobal+Geometry | 0.441667 | 0.424037 | 0.820337 |
| Mixed5 GT-TrueLogP Oracle | 0.687599 | 0.671197 | 0.869345 |
| Mixed5 GT-Margin/AnyCorrect | 0.720040 | 0.709917 | 0.850496 |

## Ranking diagnostics

| Branch | MAE | RMSE | Candidate Spearman | Within-context mean | Top-1 overlap | Top-3 overlap |
|---|---:|---:|---:|---:|---:|---:|
| GeometryOnly-Mixed5 | 0.795132 | 1.012072 | 0.191416 | 0.233173 | 0.245734 | 0.606845 |
| Prefix5+Geometry | 0.737226 | 0.980611 | 0.332012 | 0.244338 | 0.253274 | 0.620139 |
| Prefix5+RGBGlobal+Geometry | 0.721431 | 0.962161 | 0.388644 | 0.257824 | 0.260714 | 0.624206 |

Best branch: **Prefix5+RGBGlobal+Geometry** (0.441667 Acc, 0.424037 Macro-F1).
Gain recovery versus Random-ShortPrefix5: 0.287848; decision threshold: **KILL** (kill <0.52, borderline 0.52–<0.55, keep ≥0.55, strong keep ≥0.60).
Prefix contribution is assessed by the Prefix5+Geometry shuffle diagnostic; RGB contribution by the Prefix5+RGBGlobal+Geometry shuffle diagnostic.
High-occlusion subset is the bottom tertile of frame-0 Stay SceneVisibility (3271 contexts).
The old frame-0 RGBGlobal Visibility (.494841) and Frame0 Task+VisibilityAux (.506151) values are different-protocol references only.

No follow-up experiment was started automatically.

```text
policy_test_used=false
training_split=Policy Train
evaluation_split=Moving Val
prefix5_mixed_protocol=true
future_candidate_observation_used=false
gt_action_used_for_selector=false
frozen_recognizer=true
```
