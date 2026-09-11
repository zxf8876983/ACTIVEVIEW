# Reduced12 Segment-aware Sequential Recognition Audit

Train contexts: 46324; Val Moving contexts: 10080.
Every five-frame chunk is encoded independently by one shared chunk encoder. No cross-view raw skeleton sequence is constructed; history is fused only in posterior or feature space. View changes are a discrete-time 1-hop approximation, not continuous robot motion.

## Fixed-view cumulative recognition

| Fusion | t5 | t10 | t15 | t20 | t25 | t30 |
|---|---:|---:|---:|---:|---:|---:|
| MeanLogP | 0.071825 | 0.076885 | 0.091964 | 0.094940 | 0.099008 | 0.092956 |
| MeanFeature | 0.232341 | 0.263095 | 0.288591 | 0.302381 | 0.311310 | 0.311210 |

The previous raw-stitch PrefixHAR reference is preserved under `raw_stitch_prefix_reference` in fixed_view_progression.json for direct comparison.
The previous Raw-Stitch Oracle reference at t30 is 0.462996/0.503294; it is recorded explicitly in result.json.

## Sequential t30 comparison

| Method | Accuracy | Macro-F1 | Move rate |
|---|---:|---:|---:|
| MeanLogP/Stay | 0.304861 | 0.343727 | 0.000000 |
| MeanLogP/Random-1Hop | 0.351389 | 0.395866 | 0.368075 |
| MeanLogP/Privileged-Greedy-Oracle | 0.489583 | 0.538198 | 0.290437 |
| MeanFeature/Stay | 0.311210 | 0.337410 | 0.000000 |
| MeanFeature/Random-1Hop | 0.358631 | 0.390655 | 0.368075 |
| MeanFeature/Privileged-Greedy-Oracle | 0.487798 | 0.527362 | 0.290813 |
MeanLogP SegmentOracle - Stay = 18.47pp; retained fraction of previous +14.34pp = 128.8%.
MeanFeature SegmentOracle - Stay = 17.66pp; retained fraction of previous +14.34pp = 123.1%.

## Temporal complementarity

Counts and fractions for current-chunk error corrected by history, and current-chunk correctness lost after fusion, are in temporal_complementarity.json.

## View-switch drift sanity

Feature/log-probability transition deltas for same-view and switched-view transitions are in feature_transition_audit.json. They are diagnostic only and are never used to select a viewpoint.

## Answers

1. Without raw stitching, MeanFeature rises from 0.232341 at t5 to 0.311210 at t30 (monotonic=False); MeanLogP is lower and not monotonic (monotonic=False). Thus cumulative recognition remains possible, but feature fusion is substantially more stable than posterior averaging.
2. Segment-aware Oracle gains over Stay are reported for both fusion rules; this is the direct sequential information-gain estimate.
3. The retained fraction relative to the previous raw-stitch +14.34pp is stated for each fusion rule. Both segment-aware gains exceed the old gain, so the sequential signal is not explained by raw stitching alone.
4. MeanFeature is the more stable fusion: it is far stronger in fixed-view recognition and has a slightly higher Stay/Random/Oracle final profile than MeanLogP; MeanLogP posterior averaging is poorly calibrated for these chunk logits.
5. Temporal complementarity is present when history corrects current-chunk errors, but harm counts are also reported explicitly at t10–t30; this is an empirical trade-off rather than a guaranteed monotonic gain.
6. The segment-aware oracle retains a substantial gain, so a learned sequential NBV policy is scientifically worth considering; this audit itself does not train one.

## Flags

```text
policy_test_used=false
new_rgb_generated=false
new_skeleton_generated=false
nbv_policy_trained=false
gt_action_used_for_oracle_only=true
raw_cross_view_skeleton_stitching_used=false
continuous_robot_motion_claimed=false
```
