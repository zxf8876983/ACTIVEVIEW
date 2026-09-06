# Reduced14 Selector Bottleneck (Val)

This is a Val-only diagnostic. The formal WM-E/JR/ST-GCN artifacts were read-only; the privileged JR was trained on Train only and was not used by the formal method.

- Moving contexts: 14809
- WM-E candidate Top-1 positive hit (all contexts): 0.516780
- WM-E candidate Top-3 positive hit (all contexts): 0.681748
- Oracle-positive contexts: 12912
- WM-E Top-1/Top-3 conditional on oracle-positive: 0.592704 / 0.781908

| Selector | Positive action hit | Stay rate | Terminal Accuracy | Terminal Macro-F1 |
|---|---:|---:|---:|---:|
| Normal JR | 0.479371 | 0.304207 | 0.479371 | 0.472555 |
| Privileged JR (true recognition input) | 0.595584 | 0.108380 | 0.595584 | 0.604960 |
| SafeOracle | 0.875886 | 0.071848 | 0.875886 | 0.873737 |

WM-E Top-1 correct but normal JR wrong: 2223 (0.150111).
WM-E Top-1 wrong but normal JR correct: 1669 (0.112702).

## Interpretation

The WM-E ranking is the upstream candidate-recognition reference, while normal JR adds a learned Stay/candidate decision. A large privileged-JR improvement over normal JR with true candidate recognition supplied as a diagnostic input indicates selector/action-scoring loss; a low WM-E Top-1/Top-3 hit indicates an upstream WM-E ranking ceiling. These diagnostics do not alter the formal checkpoints or protocol.

Leakage audit: `test_used=false`; Val is never used for privileged training; true recognition is used only in the explicitly privileged diagnostic and offline terminal/positive diagnostics.
