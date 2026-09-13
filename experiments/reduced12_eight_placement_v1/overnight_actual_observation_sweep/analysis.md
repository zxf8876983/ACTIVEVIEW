# Reduced12 Overnight Actual Observation Research Sweep

Only frozen reduced12 ST-GCN/shared-head caches and Policy Train/Moving-Val artifacts were read. Policy Test, VLM, new RGB/skeleton/DINO generation and recognizer fine-tuning were not used.

## Protocol reproduction gate

Moving Val contexts: **10080**. Action set is exactly `Stay/current + Stage-A legal candidate_pool`; candidate count mean/min/max=6.816/2/21.
Stay: 0.302579 Acc / 0.292976 F1. Full legal GT-TrueLogP oracle: 0.753175 / 0.755358; AnyCorrect coverage=0.791964.
The gate reproduces the historical Stay/oracle references within the recorded cache-order protocol; no Test artifact was opened.

## Real second observation

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Random candidate only (O1) | 0.379266 | 0.380610 |
| BestObservedConfidence(O0,O1) | 0.426091 | 0.417529 |
| Posterior average(O0,O1) | 0.432044 | 0.429460 |
| MeanLogP(O0,O1) | 0.429762 | 0.424594 |
| MeanFeature(O0,O1) | 0.422718 | 0.414319 |
| GT-TrueLogP best-of-two | 0.495238 | 0.493802 |

## Observation budget

| Method | Accuracy | Macro-F1 | Mean observations |
|---|---:|---:|---:|
| Random-B1-MeanLogP | 0.302579 | 0.292976 | 1.000 |
| Random-B1-MeanFeature | 0.302579 | 0.292976 | 1.000 |
| Random-B1-BestObservedConfidence | 0.302579 | 0.292976 | 1.000 |
| Random-B2-MeanLogP | 0.429762 | 0.424594 | 2.000 |
| Random-B2-MeanFeature | 0.422718 | 0.414319 | 2.000 |
| Random-B2-BestObservedConfidence | 0.426091 | 0.417529 | 2.000 |
| Random-B3-MeanLogP | 0.487401 | 0.477211 | 3.000 |
| Random-B3-MeanFeature | 0.476885 | 0.464564 | 3.000 |
| Random-B3-BestObservedConfidence | 0.475992 | 0.465088 | 3.000 |
| Random-B4-MeanLogP | 0.521230 | 0.505236 | 3.920 |
| Random-B4-MeanFeature | 0.508433 | 0.490467 | 3.920 |
| Random-B4-BestObservedConfidence | 0.499901 | 0.488083 | 3.920 |

## Complementarity and stop/continue

Random-B2 MeanLogP gain over Stay: +12.72pp; Random-B3 gain: +18.48pp.
NeedMove verifier AUROC/AUPRC: 0.657081/0.598345. The threshold policy is reported only when the AUROC gate is met; it does not use future evidence before acquisition.
Observed candidate confidence/entropy/margin diagnostics are post-acquisition evidence and are not unseen-candidate predictors.

## Scientific answers

A. A second real observation is evaluated as a finite-observation budget abstraction: each full candidate observation is independently acquired; this report does not claim continuous synchronized navigation.
B. B=2 and B=3 random fusion gains are +12.01pp and +17.43pp for MeanFeature.
C. The privileged unrestricted B4 reference reaches 0.680060 Acc; the preregistered sweep decision is **STRONG KEEP finite-observation active perception**.
D. Complementarity, per-class and occlusion-stratified counts are retained in their dedicated JSON files; rescue and harm are reported rather than hidden.
E. No pre-action unseen-candidate utility predictor or new policy was trained. The only Train-only model is the small current-observation verifier.

## Flags

```text
policy_test_used=false
training_used=true_for_current_observation_verifier_only
new_rgb_generated=false
new_skeleton_generated=false
new_dino_generated=false
frozen_stgcn_modified=false
future_candidate_observation_read_before_selection=0
gt_label_used_for_privileged_oracle_only=true
observation_budget_is_discrete_full_view_abstraction=true
```
