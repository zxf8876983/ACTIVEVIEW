# Reduced12 Multi-view Fusion Ceiling Audit

Moving Val contexts: 10080; legal candidate samples: 68702.
Policy Test was not read.  This is a privileged GT-label-conditioned ceiling, not a deployable method.

## Reference and unrestricted fusion

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| S0-only | 0.254266 | 0.235500 |
| FrozenStageCv0 | 0.454266 | 0.444782 |
| BestSingle Oracle | 0.709623 | 0.704598 |
| AnyCorrect-single-view Oracle | 0.709623 | 0.704598 |
| MeanLogP-B1 | 0.709623 | 0.704598 |
| MeanLogP-B2 | 0.740575 | 0.736245 |
| MeanLogP-B3 | 0.745139 | 0.741971 |
| MeanLogP-B4 | 0.745734 | 0.742497 |
| SumLogP-B1 | 0.709623 | 0.704598 |
| SumLogP-B2 | 0.740575 | 0.735731 |
| SumLogP-B3 | 0.745139 | 0.740670 |
| SumLogP-B4 | 0.745734 | 0.741224 |

For a fixed cardinality, MeanLogP and SumLogP induce the same margin ordering; with the requested at-most budget, their different scales can select different cardinalities, so both results are retained.
MeanFeature was skipped because the frozen Val cache contains no candidate-level ST-GCN feature tensor or reusable classifier-head interface.

## Complementarity

Contexts where every single legal view is wrong: 2927.
Best B4 fusion rescues 364 of these contexts (fraction 0.036111).
At B4, contexts with at least one correct single view but a wrong fused prediction: 0 (see all budgets in complementarity.json).

## K-hop fusion

Best K2-B2 Accuracy: 0.614782; best K3-B3 Accuracy: 0.686905; unrestricted B3 Accuracy: 0.745139.
Best K4-B4 Accuracy: 0.725496; unrestricted B4 Accuracy: 0.745734.
K-hop values are privileged reachability ceilings from FrozenStageCv0 H1, not learned policies.

## Scientific answers

1. B=3/B=4 exceed BestSingle by 3.55/3.61 percentage points; the fixed decision category is **mixed evidence**.
2. Fusion-correct cases with all single views wrong are the direct evidence for multi-view complementarity; their count is reported above and in complementarity.json.
3. K3/K4 capture the fraction shown above of the unrestricted B3/B4 ceiling; K-hop fusion does not assume access outside the lattice radius.
4. No new policy, recognizer, fusion network or Test evaluation was started automatically.

## Flags

```text
policy_test_used=false
training_used=false
new_rgb_generated=false
new_skeleton_generated=false
frozen_stgcn_modified=false
gt_label_used_for_oracle_only=true
deployable=false
```
