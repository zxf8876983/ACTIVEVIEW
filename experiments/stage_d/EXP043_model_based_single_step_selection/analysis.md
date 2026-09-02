# EXP043 — Model-Based Single-Step Selection

Frozen ST-GCN evaluation on WM-B imagined observations, Val-only.  The
selector summaries below are descriptive; no selector was tuned on Val.

| Selector | Accuracy | Macro-F1 | Mean regret | P90 regret | Avg moves |
|---|---:|---:|---:|---:|---:|
| Predicted-observation label score (diagnostic) | 0.683921 | 0.625949 | 1.193079 | 4.646288 | 0.8982 |
| Predicted entropy | 0.637878 | 0.574147 | 2.004323 | 7.481076 | 1.0200 |
| Predicted top-1 confidence | 0.638307 | 0.574001 | 1.989210 | 7.419143 | 1.0236 |
| Belief cross-entropy | 0.643669 | 0.591699 | 1.516141 | 5.830432 | 0.7799 |

The first row uses the frozen row label only as an offline diagnostic score and
is not deployable; it is retained to expose the observation-model ceiling.
Test data was not read.
