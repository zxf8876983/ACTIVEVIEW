# EXP054-R1 analysis

R1 fixes only the Train/Val geometry mismatch in the matched Joint Revision.
For every frozen first-step move to `v1`, Train now rebuilds the legal graph at
`current=v1` and computes each descriptor relative to `positions[v1]`, matching
the existing Val second-step state. Recognition inputs remain archived true
ST-GCN log probabilities, and the first action, target, loss and optimizer are
unchanged.

| Method | Moving Accuracy | Moving Macro-F1 | Full Accuracy | Full Macro-F1 |
|---|---:|---:|---:|---:|
| H1 real | 0.661568 | 0.632699 | 0.685780 | 0.643640 |
| Current WM + frozen JR | 0.675529 | 0.642612 | 0.695503 | 0.649220 |
| True future + frozen JR | 0.668651 | 0.637410 | 0.690713 | 0.645941 |
| True future + matched JR-R1 | 0.676555 | 0.666292 | 0.696218 | 0.669542 |
| True-future GT oracle | 0.837918 | 0.825241 | 0.808608 | 0.786767 |

Moving current→matched-R1 deltas are +0.001026 Accuracy (+0.103pp) and
+0.023680 Macro-F1 (+2.368pp). The matched-R1 Accuracy remains 0.161363 below
the GT oracle, so most decision headroom remains. The selector changed 1,864
second actions: 349 rescued, 339 harmful, net +10. GT-oracle action agreement
rose from 0.683330 to 0.692774.

## Interpretation

This is **Case B**: the geometry-correct matched selector gives a small
Accuracy gain but a clear Macro-F1 gain, while a large oracle gap remains. The
result supports some value of future-recognition information for balanced
decision quality, but does not justify treating world-model reconstruction as
the dominant bottleneck. No Test data, new perception, Habitat rendering or H3
were used.
