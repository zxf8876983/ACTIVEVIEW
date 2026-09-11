# Adapted Recognizer Single-View Headroom Audit

Moving Val only. The frozen reduced12 ST-GCN encoder and the already trained single-view classifier head were applied independently to all 32 archived viewpoints per context. All 32 archive entries were retained exactly as requested; navigation metadata was audited but not used as a filter. No active selector, sequential fusion, policy Test, or new perception was used.

## Matched Stage-D s1

Stage-D s1: Accuracy=0.531052, Macro-F1=0.556059; expected 0.531052/0.556059. The baseline gate passed before interpreting all-view headroom.
Archive-recomputed s1: Accuracy=0.531151, Macro-F1=0.556133; cache-vs-archive prediction mismatches=1, maximum feature/logp errors=2.365e-04/8.879e-04.
Random single view: Accuracy=0.320139, Macro-F1=0.331972.

## Headroom summary

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Stage-D s1 | 0.531052 | 0.556059 |
| Random single view | 0.320139 | 0.331972 |
| BestSingle Oracle | 0.957540 | 0.960192 |
| AnyCorrect Oracle (rate) | 0.957540 | — |

BestSingle − Stage-D s1: +42.649pp Accuracy. AnyCorrect − Stage-D s1: +42.649pp.
Historical frozen-recognizer references (FrozenStageCv0≈45.43%, AnyCorrect≈72.83%) are shown only for context and are not mixed with this adapted-head protocol.

## Fixed-view generalization

Across fixed lattice viewpoints, mean Accuracy/Macro-F1=0.315885/0.327744, best=r1_a6 (0.426687), worst=r4_a5 (0.197619), Accuracy range=0.229067.
明显：s1 比 32-view 固定视角均值高至少 5pp，存在 viewpoint-distribution specialization。

## Decision

Oracle 仍≥70%，recognizer 提升后仍保留较大 viewpoint headroom，主动选视角值得继续。
The oracle is computed from the adapted head's own 32-view predictions; a high value indicates remaining single-view selection headroom, while a low value should only be interpreted as recognizer saturation when viewpoint generalization is normal.

## Flags

```text
policy_test_used=false
training_used=false
new_rgb_generated=false
new_skeleton_generated=false
single_view_only=true
frozen_stgcn_modified=false
```
