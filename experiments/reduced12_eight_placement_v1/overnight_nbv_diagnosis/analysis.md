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

## Utility source structure (Val-only follow-up)

The placement/camera convention was numerically confirmed on sampled
contexts: candidate azimuth is measured from +Z toward +X, positive placement
yaw rotates the body +Z basis toward +X, and body-relative azimuth is
`wrap(world_candidate_azimuth - placement_yaw)`. The 4×8 maps retained only
actually legal cells (68,702 candidate samples; no interpolation).

Across different scene/placement assignments of the same motion, map-level
Spearman averaged 0.268 (median 0.400), compared with 0.148 (median 0.190)
for the matched different-motion baseline. Same scene/placement maps across
all actions were weaker (0.148), but restricting to the same action raised the
Spearman mean to 0.281. Thus motion state is a stronger stable source than
scene placement alone, while action-conditioned scene effects are not
negligible.

The leave-one-sample-out additive decomposition explained 47.9% of utility
variance with motion-only effects, 10.4% with scene-placement-only effects,
and 55.9% with motion+scene effects; the remaining interaction residual was
44.1%. Motion+scene was consistently strongest by radius (72.8% explained at
1.5 m, falling to 31.3% at 3.0 m), indicating that long-range views contain
more unmodeled interaction. Class patterns are heterogeneous: `throw` had
the largest additive explained variance (64.3%), while `bend` and `touching
face` retained roughly 74% interaction residual; `knock` showed the strongest
same-scene same-action consistency (Spearman 0.669).

## Reachability and sequential oracle structure

Starting from FrozenStageCv0 H1, the candidate-only K-hop oracle reached
0.538790 / 0.529863 at K=1, 0.597222 / 0.591727 at K=2, 0.654762 / 0.650562
at K=3, and 0.688393 / 0.683594 at K=4 (Accuracy / Macro-F1). The Full
candidate oracle was 0.709623 / 0.704598; hence K=3 remained 5.49 pp below
Full Accuracy and K=4 remained 2.12 pp below. K=6 was already close at
0.708234 Accuracy. The minimum-hop distribution to any correct candidate was
45.43% at hop 0, 8.45% at hop 1, 5.84% at hop 2, 5.75% at hop 3, 3.36% at
hop 4, 2.12% at hop 5+, and 29.04% unreachable.

Privileged greedy local search improved to 0.538790 at one step,
0.554663 at two, 0.557937 at three, and 0.558234 at four—well below the
corresponding reachability ceilings. Correct candidates formed small but
nontrivial basins: among contexts with at least one correct candidate, the
mean largest component had 2.23 nodes, 1.07 correct nodes were isolated on
average, and 76.3% of correct nodes lay in the largest component. This is
consistent with a mixed picture: local continuity exists, but monotonic
greedy exploration often stops before the reachable correct basin.

Taken together, the strongest current evidence is **D: strong
motion×scene×view interaction**, with a secondary motion contribution. A
sequential embodied protocol is better motivated than another one-shot scalar
utility predictor; however, the basin statistics indicate that local search
would still need non-greedy information acquisition. These conclusions are
diagnostic only and do not alter the frozen method.
