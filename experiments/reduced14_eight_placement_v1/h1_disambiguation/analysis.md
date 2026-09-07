# Reduced14 H1 Disambiguation Potential (Val)

Each Val moving context enumerates legal real archived viewpoints from s0. The frozen history-identity encoder scores the resulting [s0, candidate] history; only the candidate selection rule changes.

Val moving contexts: 14809; candidate hypotheses: 441283. Test was not read.

## History identity after H1 selection

| Selector | Accuracy | Macro-F1 | Mean entropy | s0-error correction |
|---|---:|---:|---:|---:|
| Frozen_current_H1 | 0.482815 | 0.496501 | 1.445789 | 0.405738 |
| Random_H1 | 0.341819 | 0.354743 | 1.689171 | 0.270765 |
| Min_entropy_H1 | 0.512594 | 0.504617 | 0.734740 | 0.432696 |
| Max_margin_H1 | 0.515970 | 0.510083 | 0.772546 | 0.436157 |
| IdentityOracle_H1 | 0.899588 | 0.900413 | 1.047738 | 0.870036 |

IdentityOracle minus Frozen H1 identity Accuracy: +0.416774.
Contexts with at least one candidate predicted correctly by the history classifier: 13604 / 14809 (0.918631).
Frozen H1 selected candidate equals the Stage-C-v0 recorded s1 viewpoint for 14809 / 14809 contexts.

## Interpretation

The IdentityOracle row is a privileged upper bound because it uses the ground-truth label to choose the candidate. A large gap from Frozen H1 indicates recoverable disambiguation potential in the observed candidate views; a small gap indicates that candidate choice has limited leverage under this frozen identity representation.

Leakage audit: `test_used=false`; only Val rows, archived Val skeletons, the frozen reduced14 ST-GCN and the already-trained history-identity checkpoint were accessed. No formal checkpoint was modified.
