# Old/New Val viewpoint and oracle audit

This is a Val-only audit using the same Stage-B utility/oracle functions for both datasets.
No model was trained, no Test artifact was opened, and no Habitat/data regeneration was performed.
The old side is the frozen 16-class four-region protocol and the new side is reduced14; the table compares protocol/oracle behavior, while absolute class difficulty is not a taxonomy-matched claim.

## Unified episode-level metrics

| Metric | Old Val | New Val |
|---|---:|---:|
| s0 Accuracy | 0.412526 | 0.299640 |
| Legal-view mean Accuracy | 0.422655 | 0.298058 |
| Random legal-view Accuracy | 0.424251 | 0.295370 |
| H0 CandidateOracle | 0.833631 | 0.689198 |
| H0 SafeOracle | 0.858511 | 0.719290 |
| H0 AnyCorrect | 0.858726 | 0.720113 |
| Mean # correct legal views | 3.101451 | 2.218879 |
| Mean correct-view ratio | 0.425622 | 0.297469 |
| FixedH1-H2 SafeOracle | 0.784644 | 0.605848 |
| FixedH1-H2 AnyCorrect | 0.784644 | 0.606253 |

## Scientific answers

1. The new average single-view result is compared using the same Stage-B current/candidate definitions above; the archived-32-view metric is cache-covered moving Val only.
2. Good-view sparsity is assessed by the correct-view count and ratio distributions. AnyCorrect is reported separately and is not substituted for mean single-view accuracy.
3. H0 SafeOracle is the unified argmax over current s0 plus legal candidates, with the final class always taken from the selected real prediction.
4. FixedH1-H2 SafeOracle is a separate protocol and therefore should not be interpreted as the H0 ceiling.
5. The sampled metadata checks and cache ID checks below are the alignment audit. Direct ST-GCN recomputation was not run because the habitat PyTorch environment reported no usable CUDA device; no CPU fallback was used.
6. No evidence of an alignment error is reported by the checks; regeneration is not recommended from this audit alone.

## Leakage and runtime flags

- test_used: false
- model_training: false
- habitat_regeneration: false
- direct_stgcn_recompute: NOT RUN (CUDA unavailable in habitat environment)
