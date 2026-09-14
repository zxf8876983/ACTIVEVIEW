# Pair Complementarity Generalization Audit

- Moving Val: 10,080 contexts; Train: 46,324 contexts
- Policy Test: false; recognizer modified: false; new perception generated: false
- Recognizer: frozen reduced12 ST-GCN with frozen original/shared heads
- Action set: current/Stay + Stage-A legal candidate_pool
- Fusion: fixed normalized MeanLogP; GT label/evidence only for Train priors or privileged Val oracle evaluation
- Continuous navigation synchronization is not modeled; this is a finite observation-set audit.

## Protocol reproduction
- Random B2/B3: 0.429762/0.487401 Acc; PairMean B2/B3: 0.508234/0.557639; gate=PASS.

## Required comparisons
- AdditiveQuality B2/B3: 0.511310/0.555456 Acc; PairMean - Additive = -0.308/+0.218 pp.
- ResidualPairGreedy B2/B3: 0.373413/0.418452 Acc; Residual B3 - Random B3 = -6.895 pp.
- PairMatrix vs Q(i)+Q(j): Spearman=0.773338, Pearson=0.771953; residual mean/std=0.394839/0.200308.
- RelativeGeometryPrior B2/B3: 0.434226/0.493948; B2 gap vs PairMean=-7.401 pp.

## Rotation robustness
- Cyclic +45°/+90°/+135°/+180° B2 Acc: 0.487599, 0.485119, 0.485714, 0.480754.
- Cyclic +45°/+90°/+135°/+180° B3 Acc: 0.541270, 0.544643, 0.544841, 0.544742.
- Maximum B2 drop across all cyclic shifts: 2.748 pp.
- Maximum B3 drop across all cyclic shifts: 1.637 pp.

## Cross-recognizer
- Original-vs-Shared Train pair matrix Spearman=0.935007, Pearson=0.928057; top-10 overlap=0.400.
- Original prior → Shared B2/B3: 0.509921/0.556845; Shared prior → Original B2/B3: 0.425000/0.461310.

## Decision
- PAIR GAIN MOSTLY EXPLAINED BY SINGLE-VIEW QUALITY.
- PAIR PRIOR MOSTLY REFLECTS VIEW QUALITY / DATASET PRIOR.
- NO STRONG ABSOLUTE VIEWPOINT-ID DEPENDENCE DETECTED.
- COMPLEMENTARITY IS PARTLY STABLE ACROSS RECOGNIZERS.
- PairMeanGreedy B=3 remains justified: it reaches 0.557639 Accuracy / 0.546970 Macro-F1 and outperforms Random B3 by +7.024 pp.
- No TinySetScorer was run because the deterministic B3 branch exceeded the optional <0.55 trigger.
- This audit does not authorize a follow-up experiment; the next step must be selected explicitly.
