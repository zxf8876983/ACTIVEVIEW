# MeanLogP Fixed-view / Stay Consistency Audit

Val Moving contexts compared: 10080. Both paths use the same Stage-D s0 viewpoint and six chunks [[0, 5], [5, 10], [10, 15], [15, 20], [20, 25], [25, 30]]; no Test data was read.

## Tensor comparison

| Quantity | Max absolute error |
|---|---:|
| Skeleton | 0 |
| Chunk input | 0 |
| Encoder feature | 0 |
| Encoder logits | 0 |
| log-softmax | 0 |
| MeanLogP aggregation | 0 |
| Prediction mismatches | 0 |

Root-cause diagnosis: Before the fix, fixed-view applied axis=1 log-softmax to the 3-D (N,6,12) logits tensor, normalizing across chunk/time instead of the 12-class dimension. Stay receives a 2-D (N,12) tensor, so axis=1 was the class dimension there. After the fix, no residual difference remains in the aligned path.

## Accuracy comparison

- Previous reported fixed-view MeanLogP t30 Accuracy: 0.092956349
- Recomputed fixed-view MeanLogP t30 Accuracy: 0.304861111
- Recomputed sequential Stay MeanLogP t30 Accuracy: 0.304861111
- Final verdict: fixed-view and Stay MeanLogP are now numerically consistent.

## Root cause / scope

The audit compares the direct fixed-view sequence path and the all-view Stay path at skeleton, chunk, feature, logits, log-softmax and aggregation levels. It does not alter MeanFeature, the frozen ST-GCN, Train/Val splits or runtime artifacts.

## Flags

```text
policy_test_used=false
model_retrained=false
meanfeature_modified=false
stgcn_modified=false
train_val_split_modified=false
```
