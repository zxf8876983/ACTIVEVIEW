# Move-only Counterfactual Disambiguation Oracle

Val Moving only: 10,080 contexts; 68,702 legal move candidates. Stay is excluded and no Test data were read.

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Random-Move | 0.327083 | 0.319150 |
| FullBelief-JSD | 0.462202 | 0.439582 |
| Top3Belief-JSD | 0.450794 | 0.429598 |
| Top5Belief-JSD | 0.459028 | 0.436393 |
| MoveOnly-GTTrueLogP Oracle | 0.731647 | 0.717796 |
| MoveOnly-GTMargin Oracle | 0.731746 | 0.718888 |
| MoveOnly-AnyCorrect Oracle | 0.731746 | 0.722034 |
| Frozen-Move | NOT AVAILABLE | NOT AVAILABLE |
| CandidateSpatial-Move | NOT AVAILABLE | NOT AVAILABLE |
| DeployableAll-Move | NOT AVAILABLE | NOT AVAILABLE |

## Hypothesis support

{
  "p_gt_in_top1": 0.254265873015873,
  "p_gt_in_top2": 0.38353174603174606,
  "p_gt_in_top3": 0.48174603174603176,
  "p_gt_in_top5": 0.638095238095238,
  "contexts": 10080
}

## JSD ranking diagnostics

- **FullBelief-JSD**: candidate Spearman=0.180965, Top-1 GT-best=0.307937, mean normalized regret=0.404962.
- **Top3Belief-JSD**: candidate Spearman=0.176783, Top-1 GT-best=0.301587, mean normalized regret=0.416649.
- **Top5Belief-JSD**: candidate Spearman=0.180819, Top-1 GT-best=0.304563, mean normalized regret=0.409547.

Frozen-Move, CandidateSpatial-Move, and DeployableAll-Move were not recomputed because no per-context score cache was available and CUDA was unavailable; no CPU fallback or new model inference was run.

Selector integrity: gt_action_used_for_selector=false; predicted_action_used=false; stay_used=false; multiple_action_hypotheses_retained=true; test_used=false; training_used=false.
