# Reduced12 Sequential Decision Horizon Audit

Moving Val contexts: 10080. All methods observe six 5-frame chunks (t30); only the number of viewpoint decisions differs. View switching is a discrete-time 1-hop approximation, not continuous robot motion.

## MeanFeature final t30

| Horizon | Policy | Accuracy | Macro-F1 | Move rate | Contexts |
|---|---|---:|---:|---:|---:|
| H0 | Stay | 0.311210 | 0.320819 | 0.000000 | 10080 |
| H1 | Stay | 0.311210 | 0.320819 | 0.000000 | 10080 |
| H1 | Random-1Hop | 0.321825 | 0.338077 | 0.492163 | 10080 |
| H1 | Privileged-Greedy-Oracle | 0.398115 | 0.420356 | 0.547917 | 10080 |
| H2 | Stay | 0.311210 | 0.320819 | 0.000000 | 10080 |
| H2 | Random-1Hop | 0.335813 | 0.353610 | 0.426538 | 10080 |
| H2 | Privileged-Greedy-Oracle | 0.434325 | 0.457400 | 0.409673 | 10080 |
| Full | Stay | 0.311210 | 0.320819 | 0.000000 | 10080 |
| Full | Random-1Hop | 0.358631 | 0.378294 | 0.368075 | 10080 |
| Full | Privileged-Greedy-Oracle | 0.487798 | 0.509672 | 0.290813 | 10080 |

## MeanLogP final t30

| Horizon | Policy | Accuracy | Macro-F1 | Move rate | Contexts |
|---|---|---:|---:|---:|---:|
| H0 | Stay | 0.304861 | 0.323730 | 0.000000 | 10080 |
| H1 | Stay | 0.304861 | 0.323730 | 0.000000 | 10080 |
| H1 | Random-1Hop | 0.317361 | 0.341720 | 0.492163 | 10080 |
| H1 | Privileged-Greedy-Oracle | 0.400496 | 0.428591 | 0.555159 | 10080 |
| H2 | Stay | 0.304861 | 0.323730 | 0.000000 | 10080 |
| H2 | Random-1Hop | 0.331746 | 0.358048 | 0.426538 | 10080 |
| H2 | Privileged-Greedy-Oracle | 0.434623 | 0.462917 | 0.417758 | 10080 |
| Full | Stay | 0.304861 | 0.323730 | 0.000000 | 10080 |
| Full | Random-1Hop | 0.351389 | 0.378183 | 0.368075 | 10080 |
| Full | Privileged-Greedy-Oracle | 0.489583 | 0.517026 | 0.290437 | 10080 |

### MeanFeature horizon gains
Oracle H1 − Stay = 8.690pp; H2 − Stay = 12.312pp; Full − Stay = 17.659pp.
Fraction of Full oracle accuracy gain: H1=49.2% and H2=69.7%.
Random accuracy gains (H1/H2/Full) are 1.062pp / 2.460pp / 4.742pp.

### MeanLogP horizon gains
Oracle H1 − Stay = 9.563pp; H2 − Stay = 12.976pp; Full − Stay = 18.472pp.
Fraction of Full oracle accuracy gain: H1=51.8% and H2=70.2%.
Random accuracy gains (H1/H2/Full) are 1.250pp / 2.688pp / 4.653pp.

## Answers
1. MeanFeature H1 recovers 49.2% and H2 recovers 69.7% of the Full oracle gain. H2 is a substantial intermediate horizon and is the most reasonable first learned-policy target, while Full still retains a material additional ceiling.
2. Random-1Hop gains are reported above for each horizon; any increase should be interpreted as the value of additional random view switches, not a learned policy result.
3. MeanFeature and corrected MeanLogP are evaluated with the same trajectories and chunk boundaries; their relative horizon trends are shown directly in the two tables.
4. This is a horizon audit only. It does not train or select a policy, read Policy Test, regenerate data, or alter the recognizer.

## H0 consistency

MeanFeature H0/Stay accuracy: 0.311210317; prior fixed-view reference: 0.311210317; absolute error: 0.000e+00.
MeanLogP H0/Stay accuracy: 0.304861111; prior fixed-view reference: 0.304861111; absolute error: 0.000e+00.

## Flags

```text
policy_test_used=false
training_used=false
new_rgb_generated=false
new_skeleton_generated=false
existing_stgcn_modified=false
gt_action_used_for_oracle_only=true
continuous_robot_motion_claimed=false
raw_cross_view_skeleton_stitching_used=false
```
