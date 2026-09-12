# Causal short-prefix length oracle sweep

Split: Moving Val
Contexts: 10080
Training: none
Policy Test: false
Prefix lengths: 5,8,10,12,15
Recognizer: frozen ST-GCN + frozen shared head
Action set: Stay + Stage-A legal candidates
Mixed protocol: current[0:L] + candidate[L:30]
Stay: current[0:30]
Continuous navigation: not modeled
Protocol: discrete-time view-switch approximation

| L | Random Acc/F1 | TrueLogP Acc/F1 | Margin/AnyCorrect Acc/F1 | Drop vs FullView |
|---:|---:|---:|---:|---:|
| 5 | 0.342262/0.329111 | 0.687599/0.671197 | 0.720040/0.709917 | 0.065575 |
| 8 | 0.329861/0.315272 | 0.653571/0.633511 | 0.684325/0.671529 | 0.099603 |
| 10 | 0.328770/0.308258 | 0.633234/0.608946 | 0.664087/0.645445 | 0.119940 |
| 12 | 0.321429/0.299336 | 0.614087/0.585399 | 0.646825/0.621947 | 0.139087 |
| 15 | 0.315278/0.295595 | 0.585317/0.554597 | 0.613591/0.587340 | 0.167857 |

FullView GT-TrueLogP Oracle: 0.753175/0.755358.
L10 is below the 0.65 viability threshold.
High-occlusion best oracle prefix: L=5.
Recommendation: **KEEP longer-prefix family; longest qualifying prefix is L=8**.

The prefix trade-off is explicit: longer L adds current action evidence but leaves fewer candidate frames (30−L) for the mixed observation.
Per-class best lengths and boundary amplification are diagnostic only; no deployment policy was selected per class.

```text
policy_test_used=false
training_used=false
new_rgb_generated=false
new_skeleton_generated=false
frozen_stgcn_modified=false
gt_label_used_for_oracle_only=true
deployable=false
```
