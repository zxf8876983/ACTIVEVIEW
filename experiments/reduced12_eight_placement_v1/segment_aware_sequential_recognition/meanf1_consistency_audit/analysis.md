# Macro-F1 Consistency Audit

This audit compares the ordered predictions of the original segment-aware evaluator and the sequential horizon evaluator on the same reduced12 Moving Val contexts. No Test data, retraining, fusion change or new perception data was used.

## Prediction alignment

| Group | Contexts | y_true mismatch | y_pred mismatch |
|---|---:|---:|---:|
| MeanFeature/Stay H0 | 10080 | 0 | 0 |
| MeanFeature/Full Oracle | 10080 | 0 | 0 |
| MeanLogP/Stay H0 | 10080 | 0 | 0 |
| MeanLogP/Full Oracle | 10080 | 0 | 0 |

All four comparisons use the same ordered Moving Val contexts. With zero mismatches, the differing Macro-F1 values cannot be caused by the model, fusion, chunking or prediction path.

## Metric implementation audit

**Old evaluator:** calls the shared `classification(labels, predictions)` once after all 10,080 contexts for each method/time point.

**Horizon evaluator before the fix:** `_evaluate_path()` called the same helper separately for each 128-context group. `_merge_metrics()` then averaged those group Macro-F1 values. Macro-F1 is nonlinear, so mean(per-group F1) is not equal to F1(global confusion matrix). The helper's later confusion-matrix recomputation did not overwrite the already-averaged `macro_f1` field.

**Formal definition:** fixed 12-class confusion matrix, one-vs-rest F1 for every class (including zero-support classes with F1=0), then arithmetic mean across all 12 classes. This is equivalent to `f1_score(y_true, y_pred, labels=list(range(12)), average='macro', zero_division=0)`.

**First divergence:** horizon `_merge_metrics()` at its former weighted `macro_f1 = average(group_macro_f1)` assignment.

## Corrected values

After the minimal fix, horizon reports recompute Accuracy/Macro-F1 from the merged 12×12 confusion matrix. Accuracy is unchanged and the four values now match the original evaluator.

| Group | Accuracy | Unified Macro-F1 |
|---|---:|---:|
| MeanFeature/Stay H0 | 0.311210317460 | 0.337410100630 |
| MeanFeature/Full Oracle | 0.487797619048 | 0.527361927989 |
| MeanLogP/Stay H0 | 0.304861111111 | 0.343727018129 |
| MeanLogP/Full Oracle | 0.489583333333 | 0.538197855310 |

## Decision

The consistency issue is a reporting/evaluator bug only. The fixed horizon audit is now suitable for interpreting the H1/H2 sequential horizon results and proceeding to a first learned H2 policy, subject to the separate scientific approval already requested.

```text
policy_test_used=false
training_used=false
predictions_modified=false
fusion_modified=false
```
