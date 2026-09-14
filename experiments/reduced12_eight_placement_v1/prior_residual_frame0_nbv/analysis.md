# Static View Prior + Frame0 Residual NBV Audit

Protocol: current Frame0 RGB → exactly one action from current/Stay + Stage-A legal candidate_pool → selected real O1 alone → frozen ST-GCN + old adaptive head.
Training split: Policy Train only; evaluation/model selection split: Moving Val; Policy Test used: false.
The static prior is Q(v)=mean Train GT-Margin under the old adaptive recognizer. The residual target is R(x,v)=U(x,v)-Q(v).

## Main Moving-Val results

| Method | Accuracy | Macro-F1 | Move rate |
|---|---:|---:|---:|
| Random + old adaptive | 0.359921 | 0.380746 | 0.846032 |
| Historical selector + old adaptive | 0.523115 | 0.542017 | 0.840476 |
| StaticViewPrior + old adaptive | 0.522917 | 0.546040 | 0.896329 |
| Adaptive-aware selector + old adaptive | 0.526984 | 0.547002 | 0.910417 |
| Prior+GeometryResidual | 0.518948 | 0.543900 | 0.898313 |
| Prior+RGBResidual λ=0.5 | 0.524008 | 0.547073 | 0.911905 |
| Prior+RGBResidual λ=1.0 | 0.525496 | 0.548474 | 0.921131 |
| GT-Margin Oracle | 0.803968 | 0.820770 | 0.880159 |
| GT-TrueLogP Oracle | 0.774802 | 0.788596 | 0.887103 |

## Requested comparisons

Best residual branch: **Prior+RGBResidual λ=1.0**; gain over StaticViewPrior = +0.258pp Accuracy / +0.243pp Macro-F1; gain over InstanceOnly = -0.149pp Accuracy / +0.147pp Macro-F1.
Prior vs residual (RGB λ=1): agreement=0.853472, switches=1477 (0.146528); switched prior/residual Accuracy=0.482735/0.500339.
On switched contexts, residual gain is +0.017603 Accuracy (+1.760pp).
Switched transitions: prior-wrong/residual-correct=261, prior-correct/residual-wrong=235.
RGB residual ranking: candidate Spearman=0.161243; within-context mean/median=0.121620/0.142857.
RGB shuffle: normal=0.525496, shuffled=0.514385, drop=+1.111pp.
Residual-zero reproduces prior exactly: actions_equal=True, Accuracy=0.522917.

## High-occlusion subset

Definition: bottom tertile of current Frame0 Stay SceneVisibility (3271 contexts).
| Selector | Accuracy | Macro-F1 |
|---|---:|---:|
| StaticViewPrior + old adaptive | 0.459798 | 0.495511 |
| Adaptive-aware selector + old adaptive | 0.482116 | 0.508841 |
| Prior+GeometryResidual | 0.464078 | 0.500284 |
| Prior+RGBResidual λ=1.0 | 0.470498 | 0.503949 |
| GT-Margin Oracle | 0.710486 | 0.746504 |

## Oracle and interpretation

GT-Margin Oracle Accuracy/F1=0.803968/0.820770; AnyCorrect coverage on the full legal action set=0.803968 (candidate-only=0.785417).
Frame0 RGB provides instance-specific NBV beyond the prior only if residual selection changes actions and improves selected real-O1 recognition. The observed λ=1 switch rate is 0.146528, with net rescue=26.
Decision: **KILL PRIOR+RESIDUAL FRAME0 NBV**. This is a strict one-step diagnostic; no O0+O1 fusion, B3/B4, continuous navigation or Test evaluation was used.
Largest per-class F1 gains of the best residual branch over StaticViewPrior: sit +2.015pp, crawl +1.584pp, knock +1.202pp, stand up +0.914pp.

```text
policy_test_used=false
training_split=Policy Train
evaluation_split=Moving Val
future_candidate_observation_used_for_selector=false
future_candidate_recognizer_output_used_for_selector=false
gt_action_used_for_selector=false
terminal=selected real O1 alone
```
