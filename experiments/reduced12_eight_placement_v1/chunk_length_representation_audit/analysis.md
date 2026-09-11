# Reduced12 Chunk-Length / Segment-Representation Audit

This is a fixed-view Stay-only audit on the same Stage-C Train and Stage-D Moving Val protocol. L=5/10/15 use the frozen shared segment encoder and ordered feature concatenation followed by a small MLP. L=30 is evaluated on fixed Stay s0, while the existing FrozenStageCv0-selected s1 is reported separately as the positive-control reference. Policy Test is not read and no perception data is regenerated.

## Moving Val results

| Length | Segments | Accuracy | Macro-F1 | Representation |
|---:|---:|---:|---:|---|
| 5 | 6 | 0.328571 | 0.365435 | ordered shared five-frame encoder features + MLP |
| 10 | 3 | 0.321230 | 0.354549 | ordered shared segment encoder features + MLP |
| 15 | 2 | 0.286905 | 0.322301 | ordered shared segment encoder features + MLP |
| 30 | 1 | 0.254266 | 0.235500 | frozen full-sequence reduced12 ST-GCN on fixed Stay s0 |

The existing MeanFeature L=5 reference is Acc=0.311210, Macro-F1=0.337410; it is reproduced from the frozen MeanFeature head without retraining the recognizer.
OrderedConcat L=5→10 changes accuracy by -0.73pp; L=10→15 changes it by -3.43pp.
The L=30 frozen ST-GCN on fixed Stay s0 is Acc=0.254266, Macro-F1=0.235500. The same frozen ST-GCN on the existing FrozenStageCv0-selected s1 is Acc=0.454266, Macro-F1=0.444782; the recorded reference is Acc≈0.454266 / Macro-F1≈0.444782.

## Interpretation

The FrozenStageCv0-selected s1 positive control reproduces the recorded full-sequence reference within 0.5pp. Fixed Stay s0 is a different viewpoint protocol and therefore has the lower 0.254266 ceiling.
L=10 is not at least 2pp above L=5, so a longer isolated segment does not yet show a large recovery.
L=15 does not continue with a 2pp gain over L=10; isolated segment representation remains a likely limitation.
The fixed-view L=30 s0 control is not comparable to the recorded 0.454266 Stage-C baseline because that reference uses the selected s1 viewpoint. Segment-length attribution should therefore rely on the same-view L=5/10/15 comparison.

## Flags

```text
policy_test_used=false
new_rgb_generated=false
new_skeleton_generated=false
recognizer_backbone_modified=false
fixed_view_stay_only=true
```
