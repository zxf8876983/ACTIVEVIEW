# Reduced12 Single-view Classifier Adaptation Audit

## Matched protocol

Train uses 30580 Stage-D contexts and Moving Val uses 10080 contexts. Both use the exact cached `s1_viewpoint_id` observation from `stage_d/features/{train,val}.jsonl`; no s0, Stay, candidate, or alternate viewpoint is mixed in.

The frozen reduced12 ST-GCN checkpoint is `/home/zxf/WorkSpace/code/data/ActiveView/checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/stgcn_reduced12_no_kneel_clean_best.pth`. Its 256-D penultimate feature is the only input to the learned head. The read-only archive spot check reports feature max error 4.258e-04 and logp max error 1.939e-03.

## Classifier and training

Train-only head: `Linear(256,256) -> GELU -> Linear(256,12)` with cross-entropy, AdamW, seed 42, and at most 20 epochs. The best checkpoint is selected by Moving-Val Macro-F1 and stored outside Git.
Selected epoch: 7; best Val loss: 1.499655; best Val Macro-F1: 0.556059.

## Moving Val result

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Frozen original head | 0.454266 | 0.444782 |
| Learned single-view head | 0.531052 | 0.556059 |

Learned minus Frozen: ΔAccuracy=+7.679pp, ΔMacro-F1=+11.128pp.
Decision: 达到有效候选区间（≥53%）；本轮仅记录结果，不自动扩展。

Per-class metrics and the 12×12 confusion matrices are included in `result.json` and `per_class_metrics.json` to check that any gain is not confined to one action.

## Leakage and boundaries

Inference input is only the frozen ST-GCN `s1` feature. GT label, future candidate observation/skeleton/RGB/logits, hard predicted action, geometry, policy Test, and new perception data are not used. No active view selection or continuous fusion is evaluated.

```text
policy_test_used=false
stgcn_encoder_frozen=true
new_data_generated=false
single_view_s1_only=true
```
