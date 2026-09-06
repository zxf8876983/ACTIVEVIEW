# Pretrained History Identity → Multi-positive JR (Val)

A 540→256→128 history identity encoder was trained on Train and selected by Val identity Macro-F1. Its weights initialise two independent JR variants; WM-E, ST-GCN, taxonomy and split remain frozen.

| Method | Positive action hit | Stay rate | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| NormalMultiPositiveJR | 0.479371 | 0.304207 | 0.479371 | 0.472555 |
| HistoryAwareMultiPositiveJR | 0.486596 | 0.349315 | 0.486596 | 0.478697 |
| PretrainedFrozenJR | 0.491120 | 0.320481 | 0.491120 | 0.481416 |
| PretrainedFinetuneJR | 0.489905 | 0.338105 | 0.489905 | 0.480227 |
| PrivilegedJR | 0.595584 | 0.108380 | 0.595584 | 0.604960 |
| GTLabelPrivilegedJR | 0.872037 | 0.081504 | 0.872037 | 0.869523 |
| SafeOracle | 0.875886 | 0.071848 | 0.875886 | 0.873737 |

Standalone pretrained identity: Accuracy=0.482815, Macro-F1=0.496504.
Feature-history standalone reference: Accuracy=0.484300, Macro-F1=0.500795.
Pretrained Frozen JR identity head: Accuracy=0.482815, Macro-F1=0.496504; Finetune: Accuracy=0.492201, Macro-F1=0.501795.
Frozen minus Normal JR: ΔAccuracy=+0.011750, ΔMacro-F1=+0.008861.
Finetune minus Normal JR: ΔAccuracy=+0.010534, ΔMacro-F1=+0.007672.
Normal→Privileged Accuracy gap=+0.116213; Frozen→Privileged gap=+0.104464; Finetune→Privileged gap=+0.105679.

Interpretation: compare the pretrained identity heads with the prior Feature-history MLP and compare each JR variant with Normal Multi-positive JR. Checkpoint selection used Val only; this experiment does not make a Test claim.

Leakage audit: `split=val`, `test_used=false`, no Test path is accessed, formal WM-E/ST-GCN and old JR checkpoints are unchanged.
