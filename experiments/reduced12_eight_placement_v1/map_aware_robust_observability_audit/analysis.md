# Known-3D-Map Robust Observability NBV Audit

Experiment: Map-aware Robust Observability NBV Audit
Problem: pre-action single-step active viewpoint selection for HAR
Environment: pre-built static/quasi-static HM3D map; unknown-space exploration=false; SLAM=outside scope
Current human information: exact frame-0 H36M17 pose (privileged diagnostic)
Future human motion/action/candidate RGB/skeleton: not used for selection
Action set: Stay + Stage-A legal candidates
Recognizer: frozen reduced12 ST-GCN + frozen shared head
Policy Test: false

## Moving-Val results

| Method | Accuracy | Macro-F1 | Mean nav distance (m) | Move rate |
|---|---:|---:|---:|---:|
| Stay | 0.302579 | 0.292976 | 0.0000 | 0.000000 |
| Random | 0.365079 | 0.364567 | 2.6503 | 0.843254 |
| JointVisibility | 0.481448 | 0.477980 | 1.6408 | 0.523611 |
| DenseVisibility | 0.487897 | 0.484533 | 1.7716 | 0.562202 |
| RobustVis005 | 0.488492 | 0.484956 | 1.7437 | 0.554365 |
| RobustVis010 | 0.490476 | 0.488951 | 1.8062 | 0.578274 |
| RobustVis020 | 0.494246 | 0.492626 | 1.9297 | 0.620833 |
| RobustVisMean | 0.496627 | 0.495552 | 1.9710 | 0.636409 |
| ProjectionArea | 0.478869 | 0.473351 | 2.6449 | 0.888294 |
| FOVMargin | 0.245238 | 0.252201 | 2.7918 | 0.791171 |
| RobustComposite | 0.480258 | 0.472413 | 2.7311 | 0.878472 |
| RobustComposite-nav λ=.1 | 0.475397 | 0.465934 | 2.4644 | 0.836706 |
| RobustComposite-nav λ=.25 | 0.468353 | 0.459313 | 2.1635 | 0.775496 |
| RobustComposite-nav λ=.5 | 0.460714 | 0.454234 | 1.8191 | 0.677579 |
| MapFeature-Utility | 0.515972 | 0.516658 | 3.0342 | 0.893155 |
| ScalarVisibility+Geometry | 0.520139 | 0.520467 | 0.0000 | 0.913095 |
| GT-TrueLogP Oracle | 0.753175 | 0.755358 | 2.7559 | 0.886409 |

## Gains versus old scalar visibility

- DenseGain: -3.224 pp
- RobustGain: -3.988 pp
- MapFeatureGain: -0.417 pp

## Ranking diagnostics

| Method | Candidate Spearman | Within-context Spearman | Oracle top-1 overlap |
|---|---:|---:|---:|
| JointVisibility | 0.409922 | 0.253480 | 0.260119 |
| DenseVisibility | 0.409744 | 0.264439 | 0.265278 |
| RobustVis005 | 0.408709 | 0.261393 | 0.265179 |
| RobustVis010 | 0.409462 | 0.264543 | 0.266270 |
| RobustVis020 | 0.419357 | 0.274968 | 0.270238 |
| RobustVisMean | 0.417765 | 0.278098 | 0.271825 |
| ProjectionArea | 0.143311 | 0.216358 | 0.247619 |
| FOVMargin | -0.240140 | -0.267415 | 0.098313 |
| RobustComposite | 0.349201 | 0.231525 | 0.247222 |
| RobustComposite-nav λ=.1 | 0.351482 | 0.231134 | 0.245833 |
| RobustComposite-nav λ=.25 | 0.355703 | 0.231939 | 0.239286 |
| RobustComposite-nav λ=.5 | 0.360628 | 0.235826 | 0.234127 |
| MapFeature-Utility | 0.552578 | 0.362852 | 0.306647 |
| GT-TrueLogP Oracle | 1.000000 | 1.000000 | 0.998909 |

## Accepted frame-0 visibility consistency

| Split | Max abs diff vs structured cache | Scalar-mean max abs diff | Pass |
|---|---:|---:|---|
| train | 0.00000000 | nan | True |
| val | 0.00000000 | 0.00000000 | True |

## High-occlusion subset (3271 contexts)

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Stay | 0.102109 | 0.050838 |
| Random | 0.269337 | 0.268728 |
| JointVisibility | 0.450932 | 0.454822 |
| DenseVisibility | 0.454601 | 0.458506 |
| RobustVis005 | 0.454601 | 0.457792 |
| RobustVis010 | 0.456741 | 0.462714 |
| RobustVis020 | 0.456741 | 0.461936 |
| RobustVisMean | 0.458575 | 0.464235 |
| ProjectionArea | 0.400489 | 0.409950 |
| FOVMargin | 0.152858 | 0.132427 |
| RobustComposite | 0.439315 | 0.438941 |
| RobustComposite-nav λ=.1 | 0.436564 | 0.434890 |
| RobustComposite-nav λ=.25 | 0.433507 | 0.431758 |
| RobustComposite-nav λ=.5 | 0.435035 | 0.436126 |
| MapFeature-Utility | 0.460104 | 0.466124 |
| GT-TrueLogP Oracle | 0.638031 | 0.650935 |

## Scene-density stratification

| Density | ScalarVisibility | RobustComposite | MapFeature-Utility | Oracle |
|---|---:|---:|---:|---:|
| cluttered | 0.000000 | 0.486639 | 0.511283 | 0.733967 |
| medium | 0.000000 | 0.456930 | 0.500447 | 0.725484 |
| open | 0.000000 | 0.497170 | 0.536193 | 0.800119 |

## Scientific judgment

Decision: **KILL MAP-AWARE ROUTE** (best non-oracle Moving-Val Accuracy=0.520139; preregistered thresholds are 0.55/0.58/0.60).
The map-aware values are privileged geometry diagnostics, not deployable policies. If the decision is KILL, do not continue depth, point-cloud, top-down-map neural selectors or larger map encoders; prioritize future recognizer evidence or sequential information acquisition.

```text
policy_test_used=false
training_split=Policy Train
evaluation_split=Moving Val
current_frame0_pose_used=true
future_motion_used=false
future_candidate_observation_used=false
gt_action_used=false
deployable=false
```
