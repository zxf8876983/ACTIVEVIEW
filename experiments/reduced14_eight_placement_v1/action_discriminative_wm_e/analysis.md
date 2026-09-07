# Action-Discriminative WM-E (reduced14 Val)

Train contexts: 44248; Val moving contexts: 14809. Test was not read.

## WM-E Val diagnostics

| Metric | Value |
|---|---:|
| recognition_agreement | 0.5735308600289818 |
| skeleton_recognition_agreement | 0.2916121498614218 |
| pearson | 0.5387815345089462 |
| spearman | 0.6673488874242504 |
| feature_cosine_similarity | 0.9033562397442191 |
| belief_kl | 0.3263151048527938 |
| belief_entropy_pearson | 0.775087686021214 |
| belief_entropy_spearman | 0.7711476672389402 |
| belief_margin_pearson | 0.6157878115654096 |
| belief_margin_spearman | 0.5033352950637798 |
| top1_positive_hit | 0.5785670875818759 |
| top3_positive_hit | 0.7200351137821595 |

## H1 history identity after real selected observation

| Selector | Accuracy | Macro-F1 | Mean entropy | s0 correction |
|---|---:|---:|---:|---:|
| Frozen_current_H1 | 0.482815 | 0.496501 | 1.445789 | 0.405738 |
| Real_Min_entropy_H1 | 0.512594 | 0.504617 | 0.734740 | 0.432696 |
| Real_Max_margin_H1 | 0.515970 | 0.510083 | 0.772546 | 0.436157 |
| Action_Discriminative_WM_imagined_Min_entropy_H1 | 0.392734 | 0.395487 | 1.476925 | 0.298543 |
| Action_Discriminative_WM_imagined_Max_margin_H1 | 0.360119 | 0.368381 | 1.577129 | 0.265209 |
| IdentityOracle_H1 | 0.899588 | 0.900413 | 1.047738 | 0.870036 |
| Old_WM_imagined_Min_entropy_H1 | 0.345668 | 0.365998 | 1.656082 | 0.249636 |
| Old_WM_imagined_Max_margin_H1 | 0.327166 | 0.346189 | 1.702879 | 0.235246 |
| Ranking_aware_WM_imagined_Min_entropy_H1 | 0.395840 | 0.404250 | 1.517553 | 0.324226 |
| Ranking_aware_WM_imagined_Max_margin_H1 | 0.394220 | 0.402601 | 1.525719 | 0.321129 |

## Candidate diagnostic comparison

| WM-E | Agreement | Pearson | Spearman | Top-1 | Top-3 |
|---|---:|---:|---:|---:|---:|
| Old WM-E | 0.477298 | 0.498811 | 0.615925 | 0.516780 | 0.681748 |
| Ranking-aware WM-E | 0.519406 | 0.576882 | 0.657420 | 0.544466 | 0.694780 |
| Action-discriminative WM-E | 0.573531 | 0.538782 | 0.667349 | 0.578567 | 0.720035 |

## Interpretation

Action-discriminative WM-E imagined min-entropy Accuracy is 0.392734; the max-margin value is 0.360119.
The action heads directly supervise frozen ST-GCN feature/posterior targets and add frozen History Identity belief consistency; old WM-E pose/velocity supervision remains unchanged.
If feature/belief fidelity improves while H1 remains below Frozen H1, candidate-specific scene/occlusion context—not another selector loss—should be the next representation change.
H1 uses s0-only history against WM-E checkpoints trained with H=2 histories; the fixed interface distribution shift is recorded in result.json.
Leakage audit: `test_used=false`; no Test data were read, ST-GCN/History Identity were frozen, and the old WM-E checkpoint was not overwritten.
