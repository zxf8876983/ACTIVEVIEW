# Ranking-aware Recognition WM-E (Train/Val)

The WM-E pose/velocity objective was retained and augmented with a 14-way candidate recognition head, KL recognition loss (0.1), and fixed within-context pairwise logistic ranking loss (0.2). The old WM-E initialized the new run; its checkpoint was not modified.

Train contexts: 44248; Val moving contexts: 14809. Test was not read.

## WM-E candidate diagnostics

| Version | Agreement | Pearson | Spearman | Top-1 | Top-3 | Oracle-positive Top-1 | Oracle-positive Top-3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Old WM-E | 0.477298 | 0.498811 | 0.615925 | 0.516780 | 0.681748 | 0.592704 | 0.781908 |
| Ranking-aware head | 0.519406 | 0.576882 | 0.657420 | 0.544466 | 0.694780 | 0.624458 | 0.796856 |

## JR comparison

| Method | Positive action hit | Stay rate | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| NormalMultiPositiveJR | 0.479371 | 0.304207 | 0.479371 | 0.472555 |
| PretrainedFrozenJR_old_WM_E | 0.491120 | 0.320481 | 0.491120 | 0.481416 |
| PretrainedFrozenJR_ranking_aware_WM_E | 0.492538 | 0.226822 | 0.492538 | 0.488929 |
| PrivilegedJR | 0.595584 | 0.108380 | 0.595584 | 0.604960 |
| GTLabelPrivilegedJR | 0.872037 | 0.081504 | 0.872037 | 0.869523 |
| SafeOracle | 0.875886 | 0.071848 | 0.875886 | 0.873737 |

Ranking-aware WM-E minus old WM-E for Pretrained-Frozen JR: ΔAccuracy=+0.001418, ΔMacro-F1=+0.007513.
Ranking-aware WM-E candidate ranking changes: ΔSpearman=+0.041495, ΔTop-1=+0.027686.
The new JR was retrained with the same Pretrained-Frozen history identity architecture and objective; only the imagined candidate recognition cache changed. The Val terminal metrics are checkpoint-selected on Val and should be interpreted as Train/Val evidence, not Test evidence.

Leakage audit: `test_used=false`, no Test path was accessed, no old WM-E/JR/ST-GCN checkpoint was overwritten, and no future observation beyond the existing cache protocol was introduced.
