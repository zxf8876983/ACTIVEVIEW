# Reduced12 overnight NBV diagnosis

This synthesis uses only the reduced12 eight-placement policy Train/Val
artifacts. Policy Test was not read; no RGB, skeleton, DINO or perception
data was regenerated, and the frozen ST-GCN was not modified.

## Existing and tonight's reference results

| Reference | Moving Val Accuracy | Macro-F1 |
|---|---:|---:|
| FrozenStageCv0 | 0.454266 | 0.444782 |
| SceneVisibility | 0.469544 | 0.456516 |
| Candidate-Conditioned Spatial | 0.471528 | 0.463723 |
| DeployableAll G6 (MLP listwise) | 0.476290 | 0.469256 |
| AnyCorrect Oracle | 0.728274 | 0.729637 |
| Full GT-margin Oracle (first-step) | 0.728274 | 0.722059 |

Tonight's direct results:

| Diagnostic | Result |
|---|---|
| 1-hop GT-margin oracle | 0.538790 / 0.529863 |
| 2-hop GT-margin oracle | 0.597222 / 0.591727 |
| Ordinary Top-3 shortlist coverage | 0.629861 |
| Azimuth-diverse Top-3 coverage | 0.629762 |
| Lattice-diverse Top-3 coverage | 0.626885 |
| BCE-StayAware transition scorer | 0.482341 / 0.468785 |
| Listwise-H2 transition scorer | 0.408333 / 0.401269 |
| Listwise S0-only ablation | 0.411012 / 0.403548 |

Values in metric cells are Accuracy / Macro-F1. The transition scorer used
only deployable s0/H1 observations and candidate geometry at inference;
future candidate evidence was used only as Train targets and terminal
evaluation.

## Q1 — Does candidate utility have spatial locality?

There is moderate locality: GT-margin Spearman is 0.466 for angular neighbors,
0.688 for radial neighbors, and 0.601 over all one-hop pairs. Correctness is
also strongly clustered (neighbor-correct probability 0.634 when the center is
correct versus 0.163 when it is wrong). However, 39.5% of contexts place the
GT-margin-best candidate more than two lattice steps from Frozen H1. The
2-hop oracle reaches 0.5972, still 13.1 percentage points below the full
0.7283 oracle. Local refinement therefore captures useful structure but cannot
recover the global best view reliably.

## Q2 — Is proposal redundancy the main Top-K problem?

No. Ordinary Top-3 candidate coverage is 0.629861. Azimuth-diverse Top-3 is
0.629762 (-0.010 pp) and lattice-diverse Top-3 is 0.626885 (-0.298 pp).
Top-5 differences are similarly negligible. Fixed diversity rules increase
pairwise separation but do not recover additional correct candidates, so no
further diversity heuristic is justified by this audit.

## Q3 — Does a real H1 make future utility easier to predict?

Not under the fixed small action-agnostic scorer. The Listwise-H2 model with a
real H1 has 0.408333 Accuracy, slightly below the S0-only control at 0.411012;
mean within-context ranking Spearman is 0.1097 versus 0.1164 for S0-only.
The BCE-StayAware branch benefits from retaining the H1 stay option (0.482341),
but this is a stay/calibration effect rather than evidence that future utility
ranking became substantially more predictable. The predefined set-level gate
(best Accuracy >= 0.50 or Real-H1 gain >= 2 pp) was not met, so the tiny
set-level ranker was skipped.

## Q4 — Next research route

The evidence favors route **E: current observable information remains
insufficient and the sensing/action protocol should be reconsidered**. Utility
has some local continuity, but global jumps are common; fixed Top-K diversity
does not help; and a real H1 did not improve action-agnostic utility ranking.
The large 0.7283 oracle ceiling is therefore not explained by a simple local,
diversity, or independent transition scorer. No further method was launched
automatically.

## Leakage and reproducibility flags

```text
policy_test_used=false
policy_train_used_for_model_training_only=true
policy_val_used_for_evaluation_only=true
future_candidate_observation_used_at_inference=false
future_candidate_skeleton_used_as_train_target_or_terminal_eval_only=true
gt_action_predicted_at_inference=false
gt_action_used_for_supervision_or_posthoc_only=true
scene_visibility_used_as_posthoc_only=true
new_rgb_generated=false
new_skeleton_generated=false
frozen_stgcn_modified=false
```
