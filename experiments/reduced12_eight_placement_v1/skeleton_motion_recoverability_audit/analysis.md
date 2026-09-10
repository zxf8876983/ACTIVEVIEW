# Skeleton Motion Recoverability / Perception-Quality Audit

Val Moving only: 10080 contexts, 68702 legal move candidates, 20 scenes. Test was not read; no training or perception regeneration was performed.

## Candidate-level quality correlations

| Cue | Quality Spearman vs GT-margin | Context mean Spearman | Context median | P(context rho>0) | P(rho>0.3) |
|---|---:|---:|---:|---:|---:|
| mean_joint_motion | 0.204341 | 0.094934 | 0.100000 | 0.543452 | 0.406448 |
| max_joint_motion | 0.186495 | 0.094769 | 0.100000 | 0.539683 | 0.409921 |
| top4_joint_motion_mean | 0.188537 | 0.092348 | 0.100000 | 0.540079 | 0.406349 |
| total_motion_energy | 0.200077 | 0.073145 | 0.028571 | 0.508333 | 0.377679 |
| mean_velocity_norm | 0.204341 | 0.094934 | 0.100000 | 0.543452 | 0.406448 |
| mean_acceleration_norm | -0.172827 | -0.076337 | -0.085714 | 0.436607 | 0.302778 |
| mean_jerk_norm | -0.168036 | -0.073587 | -0.078788 | 0.439484 | 0.303869 |
| p95_acceleration | -0.168934 | -0.081702 | -0.087912 | 0.438988 | 0.306052 |
| p95_jerk | -0.163148 | -0.053444 | -0.051316 | 0.448115 | 0.307540 |
| velocity_smoothness | -0.172827 | -0.076337 | -0.085714 | 0.436607 | 0.302778 |
| acceleration_smoothness | -0.168036 | -0.073587 | -0.078788 | 0.439484 | 0.303869 |
| mean_bone_cv | -0.197193 | -0.062832 | -0.085714 | 0.436607 | 0.299405 |
| max_bone_cv | -0.179050 | -0.045685 | -0.071429 | 0.449206 | 0.310714 |
| bone_length_temporal_error | -0.230817 | -0.073602 | -0.051961 | 0.418056 | 0.282738 |
| lr_distance_instability | -0.236133 | -0.065616 | -0.085714 | 0.414881 | 0.279861 |
| mean_body_extent | -0.219349 | -0.120753 | -0.190476 | 0.393750 | 0.263393 |
| body_extent_cv | -0.282747 | -0.141595 | -0.195604 | 0.384921 | 0.252679 |
| min_body_extent | -0.259021 | -0.114240 | -0.154196 | 0.409722 | 0.275992 |
| motion_retention | 0.063062 | 0.099467 | 0.129634 | 0.554464 | 0.413194 |
| motion_deviation | 0.134619 | 0.141801 | 0.166667 | 0.573909 | 0.419742 |
| SceneVisibility | 0.266994 | 0.212516 | 0.187201 | 0.558730 | 0.445933 |
| HumanObservability | 0.119560 | 0.081969 | 0.133333 | 0.570040 | 0.398413 |
| ProjectedArea | 0.180444 | 0.105623 | 0.184148 | 0.591567 | 0.419444 |
| PoseConfidence | 0.263027 | 0.214117 | 0.314286 | 0.637302 | 0.507341 |
| Distance | 0.175286 | 0.118513 | 0.182372 | 0.597619 | 0.424306 |

## Single-cue selector diagnostic

| Selector | Accuracy | Macro-F1 | Move rate |
|---|---:|---:|---:|
| S0-only | 0.254266 | 0.235500 | 1.0 |
| FrozenStageCv0 | 0.454266 | 0.444782 | 1.0 |
| RandomMove | 0.333829 | 0.331380 | 1.0 |
| SceneVisibility | 0.469544 | 0.456516 | 1.0 |
| MaxMotionRetention | 0.329861 | 0.314566 | 1.0 |
| MinBoneInstability | 0.225893 | 0.221297 | 1.0 |
| MinJerk | 0.236806 | 0.233818 | 1.0 |
| MaxMotionEnergy | 0.314782 | 0.296661 | 1.0 |
| ProjectedArea | 0.425496 | 0.413591 | 1.0 |
| PoseConfidence | 0.455754 | 0.438806 | 1.0 |
| AnyCorrect Oracle | 0.728274 | 0.729637 | existing |
| Candidate-Conditioned Spatial | 0.471528 | 0.463723 | existing |

## Selected vs GT-best paired audit

Move pairs available: 9299; oracle-correct/selector-wrong pairs: 2293; selected stay/missing excluded: 781.
On oracle-correct/selector-wrong pairs, GT-best has lower bone instability/jerk/body extent CV on all three cues: True; higher motion retention: False.

## Same-azimuth wrong→correct pairs

Pairs: 11800. Correct-minus-wrong cue summaries are in `same_azimuth_pairs.json`; this is a paired diagnostic, not a deployable selector.

## High/low utility artifact check

High-utility means top 20% within each context; low-utility means bottom 20%. Any high-utility increase in instability cues: True.

## Scientific answers

Q1. GT-best skeleton stability: usually improved on the oracle-correct/selector-wrong subset.
Q2. GT-best temporal motion retention: not consistently higher; this is a cross-view proxy only.
Q3. Same-azimuth wrong→correct flips show the requested skeleton-quality improvement pattern: not clearly (correct-better fractions: bone_cv=0.505, jerk=0.538, body_extent_cv=0.451).
Q4. Best temporal context-ranking cue: motion_deviation (mean Spearman 0.141801); strongest reference cue: PoseConfidence (mean Spearman 0.214117); temporal-minus-reference=-0.072316.
Q5. Most supported explanation: D. ST-GCN/perception artifact exploitation remains a concern: some high-utility candidates are less temporally stable, so utility cannot be equated with physical quality.

Motion-retention proxy is not GT motion fidelity. No MPJPE, velocity error, or acceleration error was computed because exact GT/estimated canonical alignment was not verified.

Flags: `test_used=false`; `training_used=false`; `new_rgb_rendered=false`; `new_skeleton_generated=false`; `gt_action_used_for_posthoc_diagnostic_only=true`; `estimated_skeleton_temporal_cues_used=true`; `gt_motion_fidelity_used=false`; `deployable=false.
