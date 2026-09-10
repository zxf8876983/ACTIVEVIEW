# Selected-vs-Oracle Utility Regret Audit

Val Moving only (10,080 contexts), using the unified stay + legal-candidate action set. No Test data were read, no models were trained, and no perception data were regenerated.

| Selector | Accuracy | Macro-F1 | Mean rank percentile | Median rank | P(rank<=3) | Mean regret | Median norm regret | Oracle-correct/selector-wrong | Close-call | Severe-miss | Top3 coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FrozenStageCv0 | 0.454266 | 0.444782 | 0.591941 | 3.00 | 0.579663 | 5.126070 | 0.323714 | 2762 | 0.030775 | 0.891021 | 0.635714 |
| Candidate-Conditioned Spatial | 0.471528 | 0.463723 | 0.630636 | 3.00 | 0.618849 | 4.761113 | 0.260799 | 2588 | 0.037867 | 0.885240 | 0.642163 |
| RealEvidence-GTMarginListwise | 0.502778 | 0.499828 | 0.657532 | 2.00 | 0.677183 | 4.057227 | 0.032850 | 2273 | 0.021117 | 0.930048 | 0.651488 |
| RealEvidence-CorrectnessBCE | 0.501587 | 0.482541 | 0.602240 | 2.00 | 0.617262 | 4.642776 | 0.158094 | 2285 | 0.007440 | 0.973742 | 0.637401 |
| ConfidenceWeightedConsensus | 0.473909 | 0.455187 | 0.586443 | 2.00 | 0.595635 | 4.924103 | 0.123619 | 2564 | 0.000390 | 0.993760 | 0.513988 |
| Real-GTMargin Oracle | 0.728274 | 0.722059 | 1.000000 | 1.00 | 1.000000 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | 0.728274 |

## Oracle alignment

Real-GTMargin Oracle Accuracy=0.728274; unified action set includes stay.

## Main answers

Best evaluated non-oracle selector: **RealEvidence-GTMarginListwise**, Accuracy=0.502778, mean normalized regret=0.331210, Top3 correct coverage=0.651488.
Q1: selected viewpoints are not consistently close to the GT-optimal utility; utility regret indicates substantial candidate mis-ranking.
Q2: the evidence points to coarse ranking/top-1 resolution: correct viewpoints often enter Top-3 but are not selected first.
Q3: on oracle-correct but selector-wrong contexts, Top-3 recovery is 0.651488; the subset contains 2273 contexts, with close-call fraction 0.021117 and severe-miss fraction 0.930048.

No selector was retrained or modified. This is a privileged regret audit only.

`test_used=false`; `training_used=false`; `gt_action_used_for_privileged_diagnostic_only=true`; `future_candidate_skeleton_used_for_terminal_oracle_analysis_only=true`; `deployable=false`.
