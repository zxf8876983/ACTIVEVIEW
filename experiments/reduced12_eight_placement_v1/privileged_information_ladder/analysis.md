# Privileged Information Ladder / Oracle Gap Decomposition

Protocol:
pre-action / frame-0 full-view terminal recognition

Train:
Policy Train

Val:
Moving Val

Policy Test:
false

Action set:
Stay + Stage-A legal candidates

Recognizer:
frozen ST-GCN + frozen shared head

Terminal evaluation:
selected viewpoint's real full 30-frame skeleton

GT action:
privileged diagnostic input only where explicitly stated

Real candidate Frame0SceneVisibility:
privileged diagnostic input only where explicitly stated

Candidate recognizer output:
training target / oracle only, never learned-branch inference input

Moving Val contexts: 10080; Train contexts: 46324.
Existing GeometryOnly-TrueLogP reference protocol_match=True; reference Acc/F1=0.479762/0.479889. This ladder re-runs GeometryOnly with the unified architecture used by all learned branches.

## Information ladder (Moving Val)

| Method | Information | Deployable? | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Stay | none | True | 0.302579 | 0.292976 |
| Random | legal action set | True | 0.365079 | 0.364567 |
| GeometryOnly Utility | G | True | 0.487302 | 0.488664 |
| RGBGlobal Task+VisibilityAux (historical) | RGB+G | True | 0.506151 | 0.503722 |
| Real Frame0 Visibility argmax | V | False | 0.489782 | 0.485962 |
| RealVisibility+Geometry | V+G | False | 0.520139 | 0.520467 |
| Global Viewpoint Prior | global view prior | True | 0.484921 | 0.483722 |
| GTAction+ViewpointPrior | Y+view ID | False | 0.500000 | 0.501807 |
| GTAction+Geometry | Y+G | False | 0.488889 | 0.489240 |
| GTAction+RealVisibility | Y+V | False | 0.498214 | 0.494521 |
| GTAction+RealVisibility+Geometry | Y+V+G | False | 0.526091 | 0.522288 |
| GT-TrueLogP Oracle | exact candidate utility | False | 0.753175 | 0.755358 |

## Descriptive gap quantities

These are descriptive information gains, not additive causal effects.

- VisibilityGain_over_G = +0.032837
- ActionGain_over_G = +0.001587
- ConditionalVisibilityGain = +0.037202
- ConditionalActionGain = +0.005952
- ResidualGap = +0.227083
- GTAction+Geometry - DeployableBest = -0.017262
- GTAction+Visibility+Geometry - DeployableBest = +0.019940
- Oracle - DeployableBest = +0.247024

## Utility ranking

The ranking table reports frozen target utility U(c)=log p_y(c); future recognizer output is not an inference feature.

| Branch | Candidate Spearman | Within-context Spearman | Top-1 overlap | Top-3 overlap |
|---|---:|---:|---:|---:|
| GeometryOnly Utility | 0.284006 | 0.274180 | 0.276389 | 0.634127 |
| RealVisibility+Geometry | 0.455663 | 0.344485 | 0.293155 | 0.672024 |
| GTAction+Geometry | 0.416616 | 0.274162 | 0.275992 | 0.633532 |
| GTAction+RealVisibility | 0.558192 | 0.232956 | 0.273214 | 0.656746 |
| GTAction+RealVisibility+Geometry | 0.583848 | 0.354457 | 0.298512 | 0.676389 |
| Real Frame0 Visibility argmax | 0.409922 | 0.232956 | 0.260317 | 0.659325 |
| Global Viewpoint Prior | 0.285901 | 0.269974 | 0.269742 | 0.625298 |
| GTAction+ViewpointPrior | 0.433996 | 0.284419 | 0.286409 | 0.638492 |

## High-occlusion subset

Bottom tertile of current frame-0 Stay visibility: 3271 contexts.

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Stay | 0.102109 | 0.050838 |
| Random | 0.269337 | 0.268728 |
| GeometryOnly Utility | 0.411495 | 0.426211 |
| RealVisibility+Geometry | 0.462244 | 0.470903 |
| GTAction+Geometry | 0.411495 | 0.425262 |
| GTAction+RealVisibility+Geometry | 0.467747 | 0.473725 |
| GT-TrueLogP Oracle | 0.638031 | 0.650935 |
| DeployableBest | 0.452767 | 0.458178 |

## Action-specific viewpoint preference

Train legal-action aggregation yields best viewpoint IDs [0, 6, 10, 14, 22]; these are descriptive and sparse where a viewpoint is observed.

## Main conclusion

**D. LARGE RESIDUAL REMAINS EVEN WITH ACTION+VISIBILITY; PRE-ACTION ORACLE IS NOT REALISTICALLY PREDICTABLE**

GTAction branches are not candidate deployment methods. They quantify how much of the oracle gap would become predictable if action identity were known at selection time.
The gap values should not be interpreted as a strict additive causal decomposition because action, visibility and geometry interact.
No follow-up method was started automatically.

```text
policy_test_used=false
training_split=Policy Train
evaluation_split=Moving Val
future_candidate_recognizer_output_used_as_input=false
gt_action_used_as_input_only_for_privileged_branches=true
real_candidate_visibility_used_as_input_only_for_privileged_branches=true
```
