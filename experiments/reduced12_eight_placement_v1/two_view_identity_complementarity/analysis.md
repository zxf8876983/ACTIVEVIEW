# Reduced12 two-view identity complementarity

Val moving contexts only. No model was trained, no Test artifact was read, and no perception data was regenerated.

## Classification comparison

| Method | Accuracy | Macro-F1 | ΔAccuracy vs S1 | ΔMacro-F1 vs S1 |
|---|---:|---:|---:|---:|
| S0-only | 0.254266 | 0.235500 | -20.000 | -20.928 |
| S1-only | 0.454266 | 0.444782 | +0.000 | +0.000 |
| SimpleMeanSkeleton | 0.352976 | 0.336421 | -10.129 | -10.836 |
| ScalarConfidenceSelect | 0.443849 | 0.431173 | -1.042 | -1.361 |

Prior identity-estimator references (from the existing Val audit):

| Reference | Accuracy | Macro-F1 |
|---|---:|---:|
| ST-GCN feature-only MLP | 0.531052 | 0.555232 |
| Current all-feature MLP | 0.547619 | 0.566860 |

## Complementarity audit

- S0 correct rate: 0.254266
- S1 correct rate: 0.454266
- Both correct: 0.170238
- Only S0 correct: 0.084028
- Only S1 correct: 0.284028
- Neither correct: 0.461706
- Best-of-two AnyCorrect: 0.538294
- Complementarity gain over max(S0, S1): +8.403 pp
- Mean number of correct views: 0.708532

The archive stores one confidence scalar per viewpoint, not per joint. Therefore the requested fraction of high-confidence joints coming from different views is not identifiable and is reported as null; ScalarConfidenceSelect is a whole-skeleton view-level baseline.
View-level confidence winner differs between s0/s1 in 0.933135 of contexts.

## Focus actions (Recall/F1)

| Method | bend | stumble | knock | touching face |
|---|---:|---:|---:|---:|
| S0-only | 0.292/0.188 | 0.022/0.043 | 0.063/0.106 | 0.149/0.191 |
| S1-only | 0.283/0.320 | 0.160/0.237 | 0.163/0.246 | 0.214/0.256 |
| SimpleMeanSkeleton | 0.292/0.280 | 0.041/0.075 | 0.074/0.127 | 0.208/0.192 |
| ScalarConfidenceSelect | 0.288/0.294 | 0.157/0.235 | 0.138/0.213 | 0.220/0.255 |

## Scientific interpretation

Best-of-two AnyCorrect does not exceed the prior all-feature MLP accuracy, so the two archived views alone do not establish a large unused identity ceiling.
Simple direct skeleton fusion does not improve S1-only Macro-F1; a learned multi-view encoder would be needed before attributing gains to joint complementarity.
All comparisons use the same frozen reduced12 ST-GCN and the same Val moving contexts. The fixed all-feature MLP and ST-GCN feature-only references are prior Val results, not newly evaluated with this script.

## Protocol

- Skeleton convention: existing `camera_to_gravity + root_center + torso_scale + yaw_only`; no second camera rotation.
- Confidence convention: archive `(32,)` viewpoint-level scalar; no per-joint confidence was fabricated.
- `test_used=false`; no WM/JR/ST-GCN artifact was modified.
