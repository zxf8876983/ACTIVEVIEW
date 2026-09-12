# Structured Observability Privileged Audit — Final Chance

Experiment:
final privileged structured observability audit

Protocol:
pre-action/frame-0 full-view terminal recognition

Train:
Policy Train

Val:
Moving Val

Policy Test:
false

Action set:
Stay + Stage-A legal candidates

Recognizer:
frozen ST-GCN encoder + frozen shared head

Terminal HAR:
selected viewpoint real 30-frame skeleton

Structured visibility:
exact privileged frame-0 H36M17 per-joint scene visibility

Current pose:
current viewpoint frame-0 H36M17 only

Future motion input:
false

Candidate skeleton input:
false

Candidate RGB input:
false

Candidate recognizer output:
training target / oracle only

Moving Val contexts: 10080; Train contexts: 46324.

## Moving-Val results

| Method | Information | Accuracy | Macro-F1 | Move rate |
|---|---|---:|---:|---:|
| Stay | — | 0.302579 | 0.292976 | 0.000000 |
| Random | — | 0.365079 | 0.364567 | 0.843254 |
| GeometryOnly Utility | G | 0.487302 | 0.488664 | 0.881349 |
| ScalarVisibility+Geometry | Vscalar+G | 0.520139 | 0.520467 | 0.913095 |
| StructuredVisibility17+Geometry | V17+G | 0.512401 | 0.510476 | 0.887202 |
| CurrentPose+Geometry | P0+G | 0.500000 | 0.499949 | 0.886310 |
| CurrentPose+StructuredVisibility17+Geometry | P0+V17+G | 0.522421 | 0.520503 | 0.940476 |
| GTAction+CurrentPose+StructuredVisibility17+Geometry | Y+P0+V17+G | 0.520833 | 0.518689 | 0.914286 |
| GT-TrueLogP Oracle | exact candidate utility | 0.753175 | 0.755358 | 0.886409 |

## Core gains and residual

- StructuredGain (V17+G - Vscalar+G): -0.774 pp Accuracy
- PoseGain (P0+V17+G - V17+G): +1.002 pp Accuracy
- ActionResidualGain (Y+P0+V17+G - P0+V17+G): -0.159 pp Accuracy
- Final ResidualGap (GT-TrueLogP Oracle - Y+P0+V17+G): +23.234 pp Accuracy

## Scalar-equivalence sanity check

max_abs_difference=0; mean_abs_difference=0.

## Ranking diagnostics

| Branch | Candidate Spearman | Within-context Spearman | Top-1 overlap | Top-3 overlap |
|---|---:|---:|---:|---:|
| GeometryOnly Utility | 0.284006 | 0.276861 | 0.276389 | 0.634127 |
| ScalarVisibility+Geometry | 0.455663 | 0.355600 | 0.293155 | 0.672024 |
| StructuredVisibility17+Geometry | 0.460437 | 0.351421 | 0.290377 | 0.667956 |
| CurrentPose+Geometry | 0.339872 | 0.297209 | 0.281448 | 0.643849 |
| CurrentPose+StructuredVisibility17+Geometry | 0.504771 | 0.361180 | 0.300198 | 0.675992 |
| GTAction+CurrentPose+StructuredVisibility17+Geometry | 0.572713 | 0.359393 | 0.294048 | 0.674306 |
| GT-TrueLogP Oracle | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

## High-occlusion subset

Contexts: 3271.

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Stay | 0.102109 | 0.050838 |
| Random | 0.269337 | 0.268728 |
| GeometryOnly Utility | 0.411495 | 0.426211 |
| ScalarVisibility+Geometry | 0.462244 | 0.470903 |
| StructuredVisibility17+Geometry | 0.452461 | 0.457095 |
| CurrentPose+Geometry | 0.438398 | 0.446124 |
| CurrentPose+StructuredVisibility17+Geometry | 0.461021 | 0.466542 |
| GTAction+CurrentPose+StructuredVisibility17+Geometry | 0.464690 | 0.469798 |
| GT-TrueLogP Oracle | 0.638031 | 0.650935 |

## Per-joint / per-class diagnostics

Largest structured-visibility permutation effects: right_wrist (+0.001687), spine1 (+0.001091), pelvis (+0.000992), left_knee (-0.000099), right_elbow (-0.000496).
Largest per-class structured F1 gains: crawl (+0.004918), walk (+0.002805), kick (+0.002741).

## Main conclusion

**D. KILL PRE-ACTION OBSERVABILITY SELECTOR FAMILY**

Decision: **KILL entire pre-action observability selector family**.
The structured branches are privileged diagnostics, not deployable policies. The gap quantities are descriptive and not additive causal effects.
If the kill rule is met, do not continue structured visibility predictors, per-limb visibility, RGB visibility networks, body-part heatmap predictors, larger vision encoders, or task-aware visibility fusion.

```text
test_used=false
training_new_model_used=true
train_used_only_for_frozen_stgcn_masking_importance=false
gt_action_used_for_val_oracle_only=true
gt_future_visibility_used_for_oracle_only=true
deployable=false
```
