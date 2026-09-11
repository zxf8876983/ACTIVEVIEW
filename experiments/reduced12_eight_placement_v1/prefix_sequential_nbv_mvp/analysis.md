# Reduced12 Prefix-conditioned Sequential NBV MVP

Train Stage-C contexts: 46324; Val Moving contexts: 10080.
The protocol is a discrete-time view-switch approximation: after each observed 5-frame chunk, the next chunk may use Stay or a legal lattice 1-hop neighbor. Continuous robot motion is not claimed.
Policy Test was not read; no RGB, skeleton or DINO data was regenerated.

## Prefix recognizer sanity check (fixed start viewpoint)

| Observed frames | Accuracy | Macro-F1 | Mean entropy |
|---:|---:|---:|---:|
| 5 | 0.240377 | 0.258609 | 1.634868 |
| 10 | 0.273413 | 0.296323 | 1.521760 |
| 15 | 0.298313 | 0.321639 | 1.468750 |
| 20 | 0.307143 | 0.331596 | 1.443893 |
| 25 | 0.314881 | 0.341118 | 1.429486 |
| 30 | 0.319544 | 0.354676 | 1.425826 |

The zero+mask and zero-only comparison is stored in prefix_recognition.json; the deployed sequential diagnostic uses zero+mask.

## Sequential baselines

| Method | t | Accuracy | Macro-F1 | Mean entropy | GT probability | Move rate |
|---|---:|---:|---:|---:|---:|---:|
| Stay | 5 | 0.240278 | 0.258417 | 1.634861 | 0.211020 | 0.000000 |
| Stay | 10 | 0.273313 | 0.296226 | 1.521752 | 0.245370 | 0.000000 |
| Stay | 15 | 0.298313 | 0.321639 | 1.468749 | 0.270686 | 0.000000 |
| Stay | 20 | 0.307242 | 0.331714 | 1.443897 | 0.281834 | 0.000000 |
| Stay | 25 | 0.314881 | 0.341116 | 1.429486 | 0.290925 | 0.000000 |
| Stay | 30 | 0.319643 | 0.354769 | 1.425825 | 0.294329 | 0.000000 |
| Random-1Hop | 5 | 0.240278 | 0.258417 | 1.634861 | 0.211020 | 0.000000 |
| Random-1Hop | 10 | 0.263690 | 0.282646 | 1.451205 | 0.239787 | 0.492163 |
| Random-1Hop | 15 | 0.292659 | 0.312107 | 1.362693 | 0.268625 | 0.426538 |
| Random-1Hop | 20 | 0.307738 | 0.327391 | 1.313967 | 0.286475 | 0.397354 |
| Random-1Hop | 25 | 0.319841 | 0.339906 | 1.288069 | 0.298747 | 0.379936 |
| Random-1Hop | 30 | 0.323313 | 0.355680 | 1.281693 | 0.302247 | 0.368075 |
| Privileged-Greedy-Oracle | 5 | 0.240278 | 0.258417 | 1.634861 | 0.211020 | 0.000000 |
| Privileged-Greedy-Oracle | 10 | 0.375893 | 0.406938 | 1.459304 | 0.327218 | 0.467758 |
| Privileged-Greedy-Oracle | 15 | 0.418254 | 0.454353 | 1.372404 | 0.371810 | 0.358581 |
| Privileged-Greedy-Oracle | 20 | 0.436409 | 0.471271 | 1.332202 | 0.391186 | 0.307771 |
| Privileged-Greedy-Oracle | 25 | 0.451488 | 0.485279 | 1.306750 | 0.405481 | 0.275198 |
| Privileged-Greedy-Oracle | 30 | 0.462996 | 0.503294 | 1.298877 | 0.410085 | 0.257381 |

At t=30, Privileged-Greedy-Oracle is 0.462996/0.503294; Stay is 0.319643/0.354769; ΔAccuracy=14.34pp.
The largest oracle-vs-Stay Accuracy difference occurs at t30 (14.34pp relative to Stay t30).

## Boundary continuity

跨视点边界位移明显大于同视点边界，存在 skeleton discontinuity 风险. The per-method same-view and switched-view displacement counts and means are in boundary_continuity.json; large switched-boundary displacement should be treated as a limitation of this diagnostic rather than hidden.

## Answers

1. 5/10/15-frame prefix accuracy is 0.240377/0.273413/0.298313; information accumulates monotonically, but absolute identity accuracy remains low rather than immediately deployment-ready.
2. With the explicit mask, Stay improves from t5 to t30 by 7.94pp, so cumulative history is more informative than the first chunk; the zero-only ablation is reported separately.
3. The t30 privileged oracle exceeds Stay by 14.34pp Accuracy, showing a substantial diagnostic sequential ceiling under this approximation.
4. The observation chunk with the largest oracle-vs-Stay difference is reported above.
5. 跨视点边界位移明显大于同视点边界，存在 skeleton discontinuity 风险.
6. t30 oracle 相对 Stay 的提升达到 2pp 以上，支持下一步研究 learned sequential NBV policy；本轮仍未训练 policy。

## Flags

```text
policy_test_used=false
new_rgb_generated=false
new_skeleton_generated=false
existing_stgcn_modified=false
prefix_encoder_trained_on_train_only=true
gt_action_used_for_oracle_only=true
continuous_robot_motion_claimed=false
```
