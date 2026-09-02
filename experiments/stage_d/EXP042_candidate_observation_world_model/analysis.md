# EXP042 — Candidate Observation World Model

Train/Val-only CUDA run (RTX 4090, conda `habitat`).  All variants use the
frozen perceived-skeleton targets; future candidate skeletons are targets only.

| Variant | Train loss | Val MAE | Val RMSE | Val velocity RMSE |
|---|---:|---:|---:|---:|
| WM-A | 0.025537 | 0.120761 | 0.185320 | 0.181543 |
| WM-B | 0.025405 | 0.121299 | 0.187993 | 0.185257 |
| WM-C (visited RGB cache) | 0.022301 | 0.114823 | 0.178199 | 0.176229 |

Train contexts: 29,133; Val contexts: 9,742.  Each model predicts only the
Stage-D remaining p2/p3 targets (18,648 Val predictions).  No Test data was
read and no perception or ST-GCN retraining was performed.
