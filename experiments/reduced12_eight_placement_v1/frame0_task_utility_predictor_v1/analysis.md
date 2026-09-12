# Frame-0 Task-Utility Predictor

Training split: Policy Train
Model-selection split: Moving Val
Policy Test used: false

Input:
strict current frame-0 RGB DINO global feature + candidate/current geometry

Future motion input: false
candidate observation input: false
current full-action feature input: false
GT action input: false
recognizer output input: false

GT action / candidate recognizer outputs: training supervision only
Final recognizer: frozen ST-GCN encoder + frozen shared head

## Moving-Val results

| Method | Target | Accuracy | Macro-F1 | Move rate |
|---|---|---:|---:|---:|
| Stay | — | 0.302579 | 0.292976 | 0.000000 |
| Random legal | — | 0.365079 | 0.364567 | 0.843254 |
| RGBGlobal Visibility | visibility | 0.494841 | 0.493881 | 0.653671 |
| GeometryOnly-TrueLogP | GT-TrueLogP | 0.479762 | 0.479889 | 0.824702 |
| RGBGlobal-TrueLogP | GT-TrueLogP | 0.498214 | 0.495003 | 0.754563 |
| RGBGlobal-Margin | GT-Margin | 0.499306 | 0.499761 | 0.956647 |
| RGBGlobal-TrueLogP+VisibilityAux | GT-TrueLogP | 0.506151 | 0.503722 | 0.840476 |
| Historical Route-1 + Shared | old task-aware | 0.509524 | 0.509806 | 0.972024 |
| GT-TrueLogP Oracle | privileged | 0.753175 | 0.755358 | 0.886409 |

## Utility ranking

| Branch | MAE | RMSE | Candidate Spearman | Within-context Spearman | Top-1 overlap | Top-3 overlap |
|---|---:|---:|---:|---:|---:|---:|
| GeometryOnly-TrueLogP | 0.770997 | 0.982691 | 0.243396 | 0.268852 | 0.265575 | 0.629266 |
| RGBGlobal-TrueLogP | 0.710808 | 0.937307 | 0.408855 | 0.291451 | 0.268750 | 0.641766 |
| RGBGlobal-Margin | 0.955525 | 1.345497 | 0.226117 | 0.199031 | 0.258829 | 0.629266 |
| RGBGlobal-TrueLogP+VisibilityAux | 0.704270 | 0.931137 | 0.425525 | 0.307414 | 0.278571 | 0.653671 |

## Supervision audit

Train/Val contexts: 46324 / 10080; records: 313 / 105.
Stay coverage: 1.000000 (Train), 1.000000 (Val); candidate-slot coverage: 0.313707, 0.324556.
GT-TrueLogP mean/std: -1.868691/0.998347 (Train), -1.868440/1.005991 (Val); ranges: [-10.034783, -0.004935] on Val.
Action-set viewpoint identity was checked against the recognizer and visibility caches; all target recognizer outputs are supervision/terminal-only.

## High-occlusion subset

Definition: bottom tertile of frame-0 Stay visibility (3271 contexts).

| Selector | Accuracy | Macro-F1 |
|---|---:|---:|
| Random legal | 0.269337 | 0.268728 |
| RGBGlobal Visibility | 0.464384 | 0.469584 |
| GeometryOnly-TrueLogP | 0.418221 | 0.433726 |
| RGBGlobal-TrueLogP | 0.444207 | 0.453075 |
| RGBGlobal-Margin | 0.427392 | 0.435652 |
| RGBGlobal-TrueLogP+VisibilityAux | 0.452767 | 0.458178 |
| Historical Route-1 + Shared | 0.448792 | 0.455826 |
| GT-TrueLogP Oracle | 0.638031 | 0.650935 |

## Error transitions and shortcut audit

Best task vs RGBGlobal Visibility: task-correct/visibility-wrong=626, task-wrong/visibility-correct=740, both-correct=4362, both-wrong=4352.
Best task vs Historical Route-1: new-correct/old-wrong=691, new-wrong/old-correct=657, both-correct=4445, both-wrong=4287.
On 4328 disagreements with the frame-0 visibility oracle, task-correct/visibility-wrong=740, task-wrong/visibility-correct=626, mean selected GT-TrueLogP difference=+0.086928.
RGB shuffle diagnostic: normal Accuracy=0.498214, shuffled Accuracy=0.436111, change=-0.062103.

Best new task-aware branch: **RGBGlobal-TrueLogP+VisibilityAux** (Accuracy 0.506151).
RGB contribution (RGBGlobal-TrueLogP minus GeometryOnly-TrueLogP): +0.018452 Accuracy.
Visibility auxiliary contribution: +0.007937 Accuracy.
Decision: **WEAK KEEP frame0 task-utility route**.
Next step: **retain as baseline; do not add complexity** (no automatic follow-up experiment).

High-occlusion, transition and RGB-shuffle details are stored in the companion JSON files.

```text
policy_test_used=false
training_split=Policy Train
evaluation_split=Moving Val
future_candidate_observation_used=false
gt_action_used_for_predictor=false
recognizer_output_used_for_predictor=false
```
