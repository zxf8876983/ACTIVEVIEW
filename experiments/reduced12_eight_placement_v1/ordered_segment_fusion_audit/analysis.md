# Reduced12 Ordered Segment Fusion Audit

This is a fixed-view Stay-only audit. The frozen five-frame chunk encoder is reused; only the ordered fusion classifier is trained on Stage-C Train. Stage-D Moving Val is used for evaluation and Policy Test is not read.

## Classifier

Input is the chronological concatenation `[z0; z1; z2; z3; z4; z5]` (1536-D) plus a six-dimensional observed-chunk mask (1542-D). The classifier is `Linear(1542, 256) -> GELU -> Linear(256, 12)`. Unobserved chunks are zero-filled and their mask entries are zero.

## Moving Val prefix results

| Prefix | MeanFeature Acc | MeanFeature F1 | OrderedConcat Acc | OrderedConcat F1 | ΔAcc | ΔF1 |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.232341 | 0.249602 | 0.217262 | 0.238866 | -0.015079 | -0.010736 |
| 10 | 0.263095 | 0.283885 | 0.254563 | 0.278100 | -0.008532 | -0.005785 |
| 15 | 0.288591 | 0.309424 | 0.272817 | 0.305585 | -0.015774 | -0.003838 |
| 20 | 0.302381 | 0.324499 | 0.298313 | 0.328482 | -0.004067 | +0.003983 |
| 25 | 0.311310 | 0.335814 | 0.307837 | 0.335809 | -0.003472 | -0.000005 |
| 30 | 0.311210 | 0.337410 | 0.318849 | 0.351316 | +0.007639 | +0.013906 |

## Temporal-order sanity

Normal order t30: Acc=0.318849, Macro-F1=0.351316. Reversed order t30: Acc=0.266766, Macro-F1=0.294032.
OrderedConcat minus MeanFeature at t30: ΔAcc=+0.76pp, ΔF1=+1.39pp. The full-sequence frozen ST-GCN reference (~0.454266 Acc) is retained only as an external reference; it is not recomputed or modified here.

## Interpretation

At t30, OrderedConcat reaches 0.318849 accuracy versus MeanFeature 0.311210.
OrderedConcat remains below 35%, indicating that temporal order alone is not the main bottleneck; the five-frame chunk representation is likely too weak.
Reversing chunk order changes accuracy by 5.21pp, providing evidence that the classifier uses temporal order.

## Flags

```text
policy_test_used=false
training_used_for_ordered_classifier=true (Stage-C Train only)
new_rgb_generated=false
new_skeleton_generated=false
recognizer_backbone_modified=false
fixed_view_stay_only=true
```
