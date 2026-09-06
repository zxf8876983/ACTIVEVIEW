# Reduced14 History Belief Fusion (Val)

Two-view history fusion uses only the observed s0/s1 recognition log-probabilities. Candidate scoring is a privileged offline diagnostic using archived true_logp; no formal checkpoint is changed.

## Belief quality

| Belief | Top-1 Accuracy | Macro-F1 | Mean entropy |
|---|---:|---:|---:|
| S1_only | 0.415085 | 0.396831 | 0.444608 |
| Mean_probability | 0.414208 | 0.383334 | 0.954397 |
| Mean_log_probability | 0.403944 | 0.364265 | 0.612590 |
| Entropy_weighted | 0.412925 | 0.382299 | 0.574798 |
| Margin_weighted | 0.410966 | 0.380311 | 0.741924 |

## Privileged-candidate selector

| Selector | Positive action hit | Stay rate | Terminal Accuracy | Terminal Macro-F1 |
|---|---:|---:|---:|---:|
| S1_only | 0.415085 | 0.982038 | 0.415085 | 0.396831 |
| Mean_probability | 0.350800 | 0.243906 | 0.350800 | 0.325204 |
| Mean_log_probability | 0.403606 | 0.230130 | 0.403606 | 0.362122 |
| Entropy_weighted | 0.410156 | 0.286380 | 0.410156 | 0.380242 |
| Margin_weighted | 0.395368 | 0.283004 | 0.395368 | 0.365985 |
| PrivilegedJR | 0.595584 | 0.108380 | 0.595584 | 0.604960 |
| GTLabelPrivilegedJR | 0.872037 | 0.081504 | 0.872037 | 0.869523 |
| SafeOracle | 0.875886 | 0.071848 | 0.875886 | 0.873737 |

Best simple fusion: S1_only; versus S1-only selector ΔAccuracy=+0.000000, ΔMacro-F1=+0.000000.
Best simple fusion remains 0.460801 Accuracy below SafeOracle.

## Interpretation

The direct fusion diagnostics test whether two observed views improve action identity before any learned selector. Here every fusion is no better than S1-only (and several are worse), so this simple test does not support training a history-belief refiner yet. The privileged JR and GT-label privileged JR rows are reused from the preceding Train-only diagnostics for comparison.

Leakage audit: `test_used=false`; no Test rows/cache were read, no training was performed, and no formal checkpoint was modified.
