# Pair Complementarity Generalization Audit

Date: 2026-09-14

## Scope

- reduced12 Train/Moving-Val only: 46,324 Train contexts and 10,080 Moving-Val contexts.
- Candidate action set: `current/Stay + Stage-A legal candidate_pool`.
- Frozen reduced12 ST-GCN/shared head; normalized MeanLogP fusion.
- No Policy Test, recognizer training, new RGB/skeleton/DINO, or checkpoint modification.

## Results

| Method | B2 Acc / Macro-F1 | B3 Acc / Macro-F1 |
|---|---:|---:|
| Random | 0.429762 / 0.424594 | 0.487401 / 0.477211 |
| RelativeGeometryPrior | 0.434226 / 0.423513 | 0.493948 / 0.482568 |
| AdditiveQuality | 0.511310 / 0.500984 | 0.555456 / 0.545044 |
| PairMeanGreedy | 0.508234 / 0.497486 | 0.557639 / 0.546970 |
| ResidualPairGreedy | 0.373413 / 0.363775 | 0.418452 / 0.403702 |
| PairMargin Oracle | 0.706647 / 0.701819 | 0.738591 / 0.737738 |

PairMean minus AdditiveQuality is -0.308 pp (B2) and +0.218 pp (B3). Residual
B3 minus Random B3 is -6.895 pp. Pair matrix versus Q-sum has
Spearman/Pearson 0.773338/0.771953; residual mean/std are 0.394839/0.200308.

Cyclic prior shifts (+45/+90/+135/+180 degrees) produce B2 accuracies
0.487599/0.485119/0.485714/0.480754 and B3 accuracies
0.541270/0.544643/0.544841/0.544742. Maximum drops are 2.748 pp (B2) and
1.637 pp (B3), below the strong absolute-viewpoint shortcut threshold.

Original-vs-Shared Train pair matrices correlate at Spearman/Pearson
0.935007/0.928057. Cross-recognizer B2/B3 are Original-prior → Shared
0.509921/0.556845 Acc and Shared-prior → Original 0.425000/0.461310 Acc.

## Decision

PairMean B=3 remains a practical empirical baseline (+7.024 pp over Random B3),
but the gain is mostly explained by single-view quality/dataset prior. This
audit does not establish a useful residual pair-complementarity policy, and
does not show strong absolute viewpoint-ID dependence. No follow-up experiment
is authorized by this audit.
