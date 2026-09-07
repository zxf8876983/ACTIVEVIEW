# Oracle Candidate Occlusion Context (reduced14, Train/Val)

Only Train/Val Stage-D contexts were read; Test was not read.

## Candidate descriptor

Static Habitat ray-cast descriptor dimension: 10. It contains LOS bits for root, pelvis, torso, head, left_upper_body, right_upper_body, left_lower_body, right_lower_body, visible ratio, and mean normalized obstruction distance.

## WM-E diagnostics

| Metric | Action-discriminative WM-E | Oracle-occlusion WM-E |
|---|---:|---:|
| recognition agreement | 0.5735308600289818 | 0.5854354544473989 |
| candidate true-class Pearson | 0.5387815345089462 | 0.5641854480353554 |
| candidate true-class Spearman | 0.6673488874242504 | 0.6870900228038909 |
| feature cosine | 0.9033562397442191 | 0.9077457527651384 |
| belief KL | 0.3263151048527938 | 0.31506862066681635 |
| belief entropy Pearson | 0.775087686021214 | 0.7849678878185661 |
| belief entropy Spearman | 0.7711476672389402 | 0.77975443959038 |
| belief margin Pearson | 0.6157878115654096 | 0.6303363670202269 |
| belief margin Spearman | 0.5033352950637798 | 0.5228638353695073 |
| Top-1 positive hit | 0.5785670875818759 | 0.5885610101965021 |
| Top-3 positive hit | 0.7200351137821595 | 0.7265851846849889 |

## Imagined H1 identity

| Selector | Accuracy | Macro-F1 | Mean entropy | s0 correction |
|---|---:|---:|---:|---:|
| Frozen_current_H1 | 0.482815 | 0.496501 | 1.445789 | 0.405738 |
| Real_Min_entropy_H1 | 0.512594 | 0.504617 | 0.734740 | 0.432696 |
| Real_Max_margin_H1 | 0.515970 | 0.510083 | 0.772546 | 0.436157 |
| IdentityOracle_H1 | 0.899588 | 0.900413 | 1.047738 | 0.870036 |
| Oracle_Occlusion_Context_WM_imagined_Min_entropy_H1 | 0.396651 | 0.400672 | 1.453619 | 0.308561 |
| Oracle_Occlusion_Context_WM_imagined_Max_margin_H1 | 0.352623 | 0.367313 | 1.603532 | 0.255556 |
| Action_Discriminative_WM_imagined_Min_entropy_H1 | 0.392734 | 0.395487 | 1.476925 | 0.298543 |
| Action_Discriminative_WM_imagined_Max_margin_H1 | 0.360119 | 0.368381 | 1.577129 | 0.265209 |
| Old_WM_imagined_Min_entropy_H1 | 0.345668 | 0.365998 | 1.656082 | 0.249636 |
| Old_WM_imagined_Max_margin_H1 | 0.327166 | 0.346189 | 1.702879 | 0.235246 |
| Ranking_aware_WM_imagined_Min_entropy_H1 | 0.395840 | 0.404250 | 1.517553 | 0.324226 |
| Ranking_aware_WM_imagined_Max_margin_H1 | 0.394220 | 0.402601 | 1.525719 | 0.321129 |

## Interpretation

Oracle occlusion context changed candidate recognition agreement from 0.5735308600289818 to 0.5854354544473989, and candidate true-class Spearman from 0.6673488874242504 to 0.6870900228038909.
The imagined H1 min-entropy Accuracy is 0.39665068539401716; max-margin Accuracy is 0.3526234046863394.
This is a privileged static-geometry diagnostic: no action labels, recognition outputs, or real candidate predictions enter the occlusion descriptor.
If candidate fidelity and imagined H1 improve substantially, candidate-specific scene observability is a likely information gap; otherwise static occlusion alone is insufficient.

Leakage audit: test_used=false; no Test rows/cache were read; formal WM-E, JR, ST-GCN and taxonomy artifacts were not overwritten.
