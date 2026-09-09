# Reduced12 human self-visibility oracle

This is a privileged Val-only diagnostic. Human-only depth renders compare a complete
humanoid against isolated URDF body-part references; no skeleton projection heuristic is used.

## Moving Val metrics

| Method | Accuracy | Macro-F1 | Move rate |
| --- | ---: | ---: | ---: |
| S0-only | 0.254266 | 0.235500 | 0.000000 |
| FrozenStageCv0 | 0.454266 | 0.444782 | 1.000000 |
| SceneVisibility | 0.469544 | 0.456516 | 1.000000 |
| HumanSelfVisibility | 0.262302 | 0.256464 | 1.000000 |
| TotalVisibility | 0.451587 | 0.440767 | 1.000000 |
| AnyCorrect Oracle | 0.728274 | 0.729637 | 0.474008 |

HumanSelfVisibility minus FrozenStageCv0: -19.196 pp accuracy, -18.832 pp Macro-F1.
TotalVisibility uses the requested 0.5 normalized SceneVisibility - 0.5 normalized HumanSelfVisibility definition.

## Capability and interpretation

Habitat OBJECT_ID/SEMANTIC_ID collapses articulated links into the humanoid object. The
diagnostic therefore uses actual human-only depth with isolated body-part URDF renders
as reference masks. This measures self-occlusion, while excluding static-scene occlusion.
No future RGB, estimated skeleton, DINO, training, or Test data was used.

If visibility-only methods remain near the Frozen baseline, body-part visibility alone does
not explain the AnyCorrect gap; further heuristic tuning is not justified.

## Candidate diagnostics

Candidate-level Spearman(HumanSelfVisibility, GT-class true logp): -0.057037.
Candidate-level Spearman(TotalVisibility, GT-class true logp): 0.219883.
Mean HumanSelfVisibility for correct versus wrong recognizer candidates: 0.176153 versus 0.187638.
HumanSelfVisibility had no within-context discrimination in 0.00% of contexts.

## Body-part mapping summary

| Body part | Mean visibility | Spearman with GT logp |
| --- | ---: | ---: |
| head | 0.088126 | -0.035000 |
| torso | 0.012462 | -0.027514 |
| left_upper_arm | 0.208741 | -0.020766 |
| left_forearm | 0.478286 | -0.051715 |
| right_upper_arm | 0.228917 | -0.000421 |
| right_forearm | 0.366134 | -0.041963 |
| left_thigh | 0.016689 | 0.031790 |
| right_thigh | 0.017065 | -0.031595 |
| left_lower_leg | 0.215702 | -0.077425 |
| right_lower_leg | 0.205846 | -0.060601 |

## Scientific conclusion

SceneVisibility is +1.528 pp versus FrozenStageCv0, while HumanSelfVisibility is -19.196 pp and TotalVisibility is -0.268 pp.
The true body-part visibility oracle therefore does not reach the approximately 55% target and does not explain most of the AnyCorrect gap in this protocol. Further visibility-only heuristic tuning is not warranted; the next diagnostic should address view-dependent human observability or self-occlusion only if a separate scientific question requires it.
