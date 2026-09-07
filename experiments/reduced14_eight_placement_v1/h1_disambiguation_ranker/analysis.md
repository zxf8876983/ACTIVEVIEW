# Reduced14 H1 Disambiguation Ranker (Train/Val)

The ranker sees only the frozen s0 ST-GCN feature/posterior and each candidate's 9-D relative geometry. Candidate utility targets are frozen history-identity discrimination margins computed from real archived observations and are used only during Train supervision.

Train contexts: 44248; Val moving contexts: 14809. Test was not read.

## Val history identity after H1 selection

| Selector | Accuracy | Macro-F1 | Mean entropy | s0-error correction |
|---|---:|---:|---:|---:|
| Frozen_current_H1 | 0.482815 | 0.496501 | 1.445789 | 0.405738 |
| Random_H1 | 0.341819 | 0.354743 | 1.689171 | 0.270765 |
| Min_entropy_H1 | 0.512594 | 0.504617 | 0.734740 | 0.432696 |
| Max_margin_H1 | 0.515970 | 0.510083 | 0.772546 | 0.436157 |
| Deployable_Disambiguation_Ranker | 0.418732 | 0.428180 | 1.494344 | 0.326503 |
| IdentityOracle_H1 | 0.899588 | 0.900413 | 1.047738 | 0.870036 |

## Ranker diagnostics

Utility Pearson: -0.039408; utility Spearman: -0.028549; Top-1 oracle-positive hit: 0.063948.
Best checkpoint epoch: 2 (selected by Val history-identity Macro-F1).

## Interpretation

Deployable ranker versus Frozen H1: Accuracy -0.064083; Macro-F1 -0.068321.
Privileged Min-entropy and Max-margin provide upper-bound candidate-choice references (0.512594 and 0.515970 Accuracy). IdentityOracle reaches 0.899588, so the remaining gap measures candidate selection difficulty under the frozen history representation.
A positive ranker correlation and improvement over Frozen H1 would support predicting disambiguation value from currently observable state and geometry; a weak or negative result would indicate that this deployable input is insufficient for information-seeking H1 selection.

Leakage audit: `test_used=false`; no Test path was loaded; ground-truth labels and frozen history-identity margins were used only to construct Train targets and offline Val references; no formal WM-E, JR or ST-GCN checkpoint was modified.
