# History-aware Multi-positive JR v1 (Val)

The JR branch adds a 540→256→128 history identity encoder and a 14-way auxiliary identity head. WM-E, ST-GCN, taxonomy, split and the existing Multi-positive JR checkpoint remain frozen.

Train contexts: 44248; Val moving contexts: 14809.

| Method | Positive action hit | Stay rate | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| FrozenStageCv0 | 0.415085 | 1.000000 | 0.415085 | 0.396831 |
| NormalMultiPositiveJR | 0.479371 | 0.304207 | 0.479371 | 0.472555 |
| HistoryAwareMultiPositiveJR | 0.486596 | 0.349315 | 0.486596 | 0.478697 |
| PrivilegedJR | 0.595584 | 0.108380 | 0.595584 | 0.604960 |
| GTLabelPrivilegedJR | 0.872037 | 0.081504 | 0.872037 | 0.869523 |
| SafeOracle | 0.875886 | 0.071848 | 0.875886 | 0.873737 |

History identity head: Accuracy=0.467689, Macro-F1=0.477778.
History-aware JR minus normal JR: ΔAccuracy=+0.007225, ΔMacro-F1=+0.006142.
Normal JR→Privileged JR Accuracy gap: +0.116213; history-aware JR→Privileged JR gap: +0.108988.

Interpretation: the history identity branch is useful if it exceeds the normal JR on the frozen Val protocol and its identity head is materially above S1-only. Any improvement here is a JR-only Train/Val result; no formal Test claim is made.

Leakage audit: `test_used=false`; no Test artifact/path is accessed, Val is used only for checkpoint selection/evaluation, and the old WM-E/JR/ST-GCN artifacts are unchanged.
