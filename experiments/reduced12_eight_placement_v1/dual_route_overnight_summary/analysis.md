# Reduced12 Overnight Dual-Route Experiment

Policy Test was not read. All training used Policy Train; Moving Val (10,080 contexts) was evaluation/model selection only.

Shared recognizer: shared_multiview_head.

## Shared frozen-encoder head

| Population | Frozen Acc/F1 | Multi-view head Acc/F1 |
|---|---:|---:|
| s0_current | 0.254266/0.235500 | 0.302579/0.292976 |
| s1 | 0.454266/0.444782 | 0.507044/0.507894 |
| legal_candidates_micro | 0.334474/0.330999 | 0.375418/0.378473 |
| all32 | 0.290867/0.284591 | 0.323214/0.324290 |

## Route-1

| Selector | Accuracy | Macro-F1 | Move rate |
|---|---:|---:|---:|
| Stay/current | 0.302579 | 0.292976 | 0.000000 |
| Random legal action | 0.365079 | 0.364567 | 0.843254 |
| Nearest/lowest-cost | 0.302579 | 0.292976 | 0.000000 |
| Old Direct-Correctness | 0.496329 | 0.493225 | 0.964683 |
| Retrieval-NBV K=32 | 0.484325 | 0.487705 | 0.847718 |
| GT-TrueLogP Oracle | 0.702579 | 0.700609 | 0.892560 |
| Legal AnyCorrect Coverage | 0.771726 coverage (7779/10080) | — | — |

Retrieval K=32 support mean=13475.30; K=16/64 are robustness checks only.

## Route-2

Verifier AUROC=0.722272, AUPRC=0.606913, ECE=0.056462.

| Policy | Accuracy | Macro-F1 | Mean moves |
|---|---:|---:|---:|
| Stay only | 0.302579 | 0.292976 | 0.000 |
| Random-1 | 0.373016 | 0.377837 | 1.000 |
| Random-2 | 0.375298 | 0.378075 | 2.000 |
| Random-3 | 0.371032 | 0.372071 | 2.920 |
| Angular-Diversity always-1 | 0.401488 | 0.403427 | 1.000 |
| Angular-Diversity always-2 | 0.350893 | 0.351590 | 2.000 |
| Angular-Diversity always-3 | 0.378968 | 0.383935 | 2.920 |
| MaxConfidence over observed views | 0.500000 | 0.487981 | 2.920 |
| Verifier-best over observed views | 0.434226 | 0.423232 | 2.920 |
| Verifier stop/continue | 0.433929 | 0.421583 | 2.850 |

Route-1 decision: **KEEP** (primary K=32; kill rule Acc <48% or <= Random+1pp).
Route-2 decision: **KEEP** (AUROC threshold 0.65 and matched Random+1pp).

## Leakage and reproducibility flags

```text
policy_test_used=false
train_used_for_shared_head_and_verifier=true
moving_val_used_for_selection_and_evaluation=true
route1_future_observation_read_before_selection=0
route2_future_unobserved_observation_read=0
new_rgb_or_skeleton_generated=false
frozen_stgcn_modified=false
```
