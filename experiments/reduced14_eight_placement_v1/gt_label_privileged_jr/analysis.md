# Reduced14 GT-label Privileged JR (Val)

This is a diagnostic-only model trained on Train contexts and evaluated on Val moving contexts. It receives archived candidate true_logp plus a 14-D ground-truth label one-hot; no formal checkpoint is changed.

| Method | Positive action hit | Stay rate | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| NormalJR | 0.479371 | 0.304207 | 0.479371 | 0.472555 |
| PrivilegedJR | 0.595584 | 0.108380 | 0.595584 | 0.604960 |
| GTLabelPrivilegedJR | 0.872037 | 0.081504 | 0.872037 | 0.869523 |
| SafeOracle | 0.875886 | 0.071848 | 0.875886 | 0.873737 |

Privileged JR → GT-label Privileged JR: ΔAccuracy=+0.276454, ΔMacro-F1=+0.264562.
GT-label Privileged JR → SafeOracle gap: Accuracy=0.003849, Macro-F1=0.004214.
Normal JR → Privileged JR: ΔAccuracy=+0.116213, ΔMacro-F1=+0.132405.

## Interpretation

GT-label Privileged JR is close to SafeOracle; the dominant remaining limitation is action belief/identity inference.
The comparison is not a new formal method: archived recognition and the GT label are privileged inputs used only to locate the ceiling. A large Normal→Privileged gain indicates that candidate recognition/belief is important; a remaining GT-label→SafeOracle gap indicates selector/objective limitations after identity inference is removed.

Leakage audit: `test_used=false`; Train was used only for the diagnostic fit, Val only for evaluation, and no formal WM-E/JR/ST-GCN checkpoint was modified.
