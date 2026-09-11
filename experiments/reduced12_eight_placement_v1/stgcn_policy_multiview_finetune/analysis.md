# Reduced12 Policy-Multiview ST-GCN Fine-tuning Audit

The pretrained clean/BABEL reduced12 ST-GCN was initialized from the existing checkpoint and fine-tuned end-to-end with 12-class cross entropy on Policy Train observations. Every epoch samples exactly one current-or-legal-candidate observation per unique record_id. Moving Val is evaluation/model-selection only.

## Data and sampling

Policy Train contexts=46324, unique records=313, available current+legal observations=351499. Record-balanced sampler: one uniformly sampled context then one uniformly sampled current/stay-or-legal-candidate viewpoint per record_id per epoch; each epoch samples 313 observations (1–1 per record). Moving Val contexts=10080, legal observations=68702, all-32 observations=322560.

## Matched Moving-Val results

| Population | Frozen Acc/F1 | Existing adapted-head Acc/F1 | Fine-tuned Acc/F1 | Fine-tuned ΔAcc/ΔF1 vs Frozen |
|---|---:|---:|---:|---:|
| s0_current | 0.254266/0.235500 | 0.285714/0.293422 | 0.270337/0.281765 | +1.607pp/+4.627pp |
| s1 | 0.454266/0.444782 | 0.531151/0.556133 | 0.453075/0.473547 | -0.119pp/+2.876pp |
| legal_candidates_micro | 0.334474/0.330999 | 0.369815/0.392778 | 0.339452/0.365686 | +0.498pp/+3.469pp |
| all32 | 0.290858/0.284585 | 0.315885/0.330409 | 0.292699/0.307170 | +0.184pp/+2.258pp |

## Fine-tuned privileged headroom

GT-TrueLogP Oracle: 0.690278/0.703700; GT-Margin Oracle: 0.760218/0.784746; Legal AnyCorrect Coverage: 0.760218 (7663/10080).
AnyCorrect is coverage only and is not reported as Oracle Accuracy.

## Viewpoint specialization

Spearman(s1 selection frequency, FineTuned−Frozen per-view gain)=-0.47882677248190597; best gain views=r4_a6, r3_a6, r4_a7, r4_a0, r3_a3; worst gain views=r1_a4, r1_a5, r1_a3, r1_a6, r1_a0.
Fixed-view mean/best/worst accuracy: Frozen=0.290858/r1_a6 (0.372421)/r4_a5 (0.191667); Adapted=0.315885/r1_a6 (0.426687)/r4_a5 (0.197619); FineTuned=0.292699/r1_a7 (0.372520)/r4_a5 (0.190873).
FineTuned−Frozen Accuracy gains: s0=+1.607pp, s1=-0.119pp, legal=+0.498pp, all32=+0.184pp; s1−legal=-0.617pp; s1−all32=-0.303pp.

## Training and decision

Best epoch=15, train loss=2.412177, Val loss=2.578403; overfitting flag=False.
The fine-tuning does not produce a meaningful legal/all-32 gain; this route should be stopped under the predefined淘汰标准.

Per-class and per-view deltas are stored separately; no class, viewpoint, split, taxonomy, or existing checkpoint was modified.

## Flags

```text
policy_test_used=false
moving_val_used_for_model_selection_and_evaluation_only=true
training_used=policy_train_only
new_rgb_or_skeleton_generated=false
selector_or_fusion_used=false
pretrained_initialization=true
```
