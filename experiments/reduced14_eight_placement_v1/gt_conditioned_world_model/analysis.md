# GT-conditioned World Model (reduced14 Train/Val)

Train contexts: 44248; Val moving contexts: 14809. Test was not read.

The model is a privileged diagnostic initialized from the existing Action-Discriminative WM-E. A reduced14 one-hot ground-truth action is fused into the candidate condition; the old checkpoint and all formal methods remain untouched.

## WM-E Val diagnostics

| Metric | Action-Discriminative WM-E | GT-conditioned WM-E |
|---|---:|---:|
| recognition_agreement | 0.5735308600289818 | 0.592209607150729 |
| pearson | 0.5387815345089462 | 0.542777866559054 |
| spearman | 0.6673488874242504 | 0.6790590995322956 |
| feature_cosine_similarity | 0.9033562397442191 | 0.9057707897249068 |
| belief_kl | 0.3263151048527938 | 0.31704651572510656 |
| belief_entropy_pearson | 0.775087686021214 | 0.78796114296452 |
| belief_entropy_spearman | 0.7711476672389402 | 0.7885744093575834 |
| belief_margin_pearson | 0.6157878115654096 | 0.6255169845061015 |
| belief_margin_spearman | 0.5033352950637798 | 0.5308247127291223 |
| top1_positive_hit | 0.5785670875818759 | 0.587480586130056 |
| top3_positive_hit | 0.7200351137821595 | 0.7292187183469512 |

## Imagined H1 identity after real selected observation

| Selector | Accuracy | Macro-F1 | Mean entropy | s0 correction |
|---|---:|---:|---:|---:|
| Frozen_current_H1 | 0.482815 | 0.496501 | 1.445789 | 0.405738 |
| Real_Min_entropy_H1 | 0.512594 | 0.504617 | 0.734740 | 0.432696 |
| Real_Max_margin_H1 | 0.515970 | 0.510083 | 0.772546 | 0.436157 |
| Action_Discriminative_WM_imagined_Min_entropy_H1 | 0.392734 | 0.395487 | 1.476925 | 0.298543 |
| Action_Discriminative_WM_imagined_Max_margin_H1 | 0.360119 | 0.368381 | 1.577129 | 0.265209 |
| GT_conditioned_WM_imagined_Min_entropy_H1 | 0.411304 | 0.419345 | 1.469697 | 0.326412 |
| GT_conditioned_WM_imagined_Max_margin_H1 | 0.382403 | 0.394278 | 1.539284 | 0.292168 |
| IdentityOracle_H1 | 0.899588 | 0.900413 | 1.047738 | 0.870036 |

## Interpretation

GT-conditioned WM imagined H1 Accuracy is 0.4113039367951921 (min-entropy) and 0.3824025930177595 (max-margin).
A clear fidelity and H1 gain would support hypothesis-conditioned future prediction as the next method direction. If fidelity improves but imagined H1 remains below Frozen H1, the remaining limitation is in the imagined selector/target interface rather than hypothesis-agnostic future prediction.
The GT action is used only as privileged Train/Val diagnostic conditioning and is not available to a deployable policy.

Leakage audit: `test_used=false`; no Test rows, caches, or files were read. ST-GCN and History Identity were frozen, the formal WM-E/JR artifacts were not modified, and the old Action-Discriminative WM-E checkpoint was not overwritten.
