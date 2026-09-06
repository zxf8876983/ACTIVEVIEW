# Reduced14 Belief Selector Diagnostics (Val)

All four selectors use archived candidate true_logp as a privileged offline diagnostic input. No formal checkpoint or method was changed.

Moving contexts: 14809

| Method | Positive action hit | Stay rate | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| SafeOracle | 0.875886 | 0.071848 | 0.875886 | 0.873737 |
| CurrentTop1PseudoLabel | 0.415288 | 0.271389 | 0.415288 | 0.396996 |
| CurrentFullBelief | 0.415085 | 0.982038 | 0.415085 | 0.396831 |
| EntropyReduction | 0.412384 | 0.057398 | 0.412384 | 0.386942 |
| NormalJR | 0.479371 | 0.304207 | 0.479371 | 0.472555 |
| PrivilegedJR | 0.595584 | 0.108380 | 0.595584 | 0.604960 |

Full-belief minus hard Top-1 pseudo-label: ΔAccuracy=-0.000203, ΔMacro-F1=-0.000165.
Entropy reduction minus hard Top-1 pseudo-label: ΔAccuracy=-0.002904, ΔMacro-F1=-0.010054.
Privileged JR minus normal JR: ΔAccuracy=+0.116213, ΔMacro-F1=+0.132405.
Best simple selector: CurrentTop1PseudoLabel; its SafeOracle Accuracy gap is 0.460598.

## Interpretation

Current Top-1 uses the hard current pseudo-label, whereas full-belief retains uncertainty and entropy reduction targets lower recognition entropy. Here full-belief is effectively all-Stay (98.2% Stay) and does not improve over hard Top-1; entropy selection is slightly worse. Thus the current belief does not provide a useful direct score for choosing candidates. The privileged JR row is copied from the preceding Train-only privileged diagnostic, not retrained here: its +0.116213 Accuracy over normal JR shows an additional selector/objective loss when actual archived candidate recognition is supplied as a diagnostic input. The best simple selector remains 0.460598 Accuracy below SafeOracle, and privileged JR remains 0.280303 below SafeOracle, so both belief quality and selector/objective quality limit the formal method; the large privileged-vs-normal gap specifically implicates JR selection as a major remaining bottleneck.

Leakage audit: `test_used=false`; no Test rows/cache were read and no formal checkpoint was modified.
