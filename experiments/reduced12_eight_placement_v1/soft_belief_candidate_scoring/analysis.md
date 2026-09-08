# Reduced12 soft-belief candidate scoring

Val-only diagnostic. No policy Test data was read, no model was trained, and no data was regenerated.
Terminal predictions always come from the selected real archived observation passed through frozen reduced12 ST-GCN.

## Moving Val

| Method | Accuracy | Macro-F1 | Positive action hit | Stay rate |
|---|---:|---:|---:|---:|
| Current JR | 0.535615 | 0.536320 | 0.535615 | 0.372520 |
| S1Posterior + Imagined | 0.447718 | 0.438928 | 0.447718 | 0.881548 |
| LearnedBelief + Imagined | 0.503869 | 0.503919 | 0.503869 | 0.474802 |
| LearnedBelief-hard + Imagined | 0.503075 | 0.501623 | 0.503075 | 0.425694 |
| GT-onehot + Imagined | 0.678274 | 0.663142 | 0.678274 | 0.401687 |
| LearnedBelief + Real | 0.567063 | 0.583121 | 0.567063 | 0.117460 |
| LearnedBelief-hard + Real | 0.550397 | 0.565743 | 0.550397 | 0.110714 |
| GT-onehot + Real | 0.911210 | 0.908047 | 0.911210 | 0.082242 |
| FixedH1-H2 Oracle | 0.911210 | 0.908047 | 0.911210 | 0.082242 |

## Full Val

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Current JR | 0.512098 | 0.506333 |
| S1Posterior + Imagined | 0.455084 | 0.445270 |
| LearnedBelief + Imagined | 0.491506 | 0.486377 |
| LearnedBelief-hard + Imagined | 0.490991 | 0.484908 |
| GT-onehot + Imagined | 0.604633 | 0.589634 |
| LearnedBelief + Real | 0.532497 | 0.538379 |
| LearnedBelief-hard + Real | 0.521686 | 0.527340 |
| GT-onehot + Real | 0.755727 | 0.751001 |
| FixedH1-H2 Oracle | 0.755727 | 0.751001 |

## Requested comparisons (Moving Val, percentage points)

- LearnedBelief+Imagined − Current JR: **-3.175 pp Accuracy / -3.240 pp Macro-F1**.
- LearnedBelief-hard − LearnedBelief-soft (Imagined): **-0.079 pp / -0.230 pp**.
- LearnedBelief-hard − LearnedBelief-soft (Real): **-1.667 pp / -1.738 pp**.
- LearnedBelief+Real − GT+Real: **-34.415 pp / -32.493 pp**.
- GT+Imagined − GT+Real: **-23.294 pp / -24.490 pp**.

## Scientific interpretation

Direct soft-belief scoring changes Current JR by only -3.175 pp Accuracy, so it does not show a large belief-selector interface gain. Soft belief is at least as effective as hard belief on the imagined branch. LearnedBelief + Real remains 34.415 pp below GT-onehot + Real, indicating that the learned identity estimator remains a major bottleneck. The GT imagined-to-real gap is 23.294 pp Accuracy, quantifying the remaining candidate-evidence/WM gap.

## Protocol

- taxonomy: reduced12 (walk, sit, stand up, bend, crawl, stumble, clap, throw, kick, knock, punch, touching face)
- candidate budget: ALL_LEGAL; stay plus every dynamically legal H2 candidate
- score: sum_y belief[y] * exp(candidate_logp[y])
- `test_used=false`; no Test paths or artifacts were read
- no training, checkpoint modification, or data regeneration
