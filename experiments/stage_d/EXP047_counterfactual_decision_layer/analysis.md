# EXP047

Frozen imagined-recognition decision layers (Val, 13,987 trajectories):

| Method | Accuracy | Macro-F1 | Mean regret | P90 regret |
|---|---:|---:|---:|---:|
| CF_POSTERIOR | 0.618360 | 0.553563 | 1.804145 | 6.634580 |
| CF_CORRECTNESS_LOGISTIC | 0.653249 | 0.605910 | 1.441406 | 5.607818 |
| CF_CORRECTNESS_MLP | **0.664403** | **0.616854** | 1.502970 | 5.923024 |
| CF_CORRECTNESS_X_RELIABILITY | 0.661686 | 0.612595 | 1.462535 | 5.763711 |
| CF_PAIRWISE_CORRECTNESS | 0.660971 | 0.613673 | 1.507218 | 5.933156 |
| Fixed-first Oracle | 0.771502 | 0.725081 | 0.586204 | 1.699901 |

The best legal model by Accuracy/Macro-F1 was CF_CORRECTNESS_MLP.  It exceeded
EXP014 Accuracy by 0.015300 (paired bootstrap 95% CI [0.010224, 0.020519],
McNemar p=8.97e-09), but remained far above the fixed-first oracle regret.
All legal methods used imagined WM-E evidence only; privileged true-observation
and GT-label controls are non-deployable.  Full metrics and losses are in
`result.json`, `model_comparison.json`, and the compact
`selector_comparison.json`.  The learned checkpoints are local runtime
artifacts and are intentionally not part of the Git commit.
