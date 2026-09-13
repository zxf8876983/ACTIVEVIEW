# Zero-shot VLM NBV Viability Audit

```text
Subset: 1000 fixed Moving-Val contexts
Seed: 42
Training: none
VLM fine-tuning: false
Policy Test: false
Problem: pre-action single-step NBV for HAR
Future action: unknown
Known static Habitat map: true
Current exact frame0 pose: privileged viability input
Future human motion: false
Candidate real RGB: false
Candidate real skeleton: false
Candidate recognizer output: false
Action set: Stay + Stage-A legal candidates
Terminal HAR: selected real viewpoint full 30-frame skeleton -> frozen ST-GCN + shared head
```

## Matched subset results

| Method | Acc | Macro-F1 | Move rate | Mean nav (m) |
|---|---:|---:|---:|---:|
| Stay | 0.267000 | 0.262069 | 0.0000 | 0.000 |
| Random | 0.369000 | 0.374147 | 0.8370 | 2.752 |
| GeometryOnly Utility | 0.489000 | 0.489270 | 0.8930 | 3.026 |
| ScalarVisibility+Geometry | 0.510000 | 0.504596 | 0.9310 | 3.010 |
| MapFeature-Utility | 0.524000 | 0.521245 | 0.9120 | 3.268 |
| GT-TrueLogP Oracle | 0.764000 | 0.763811 | 0.8980 | 2.892 |
| VLM-RGB | 0.472000 | 0.473142 | 0.9540 | 2.794 |
| VLM-Map | 0.485000 | 0.487619 | 0.9910 | 2.961 |
| VLM-RGB+Map | 0.485000 | 0.488016 | 0.9910 | 2.942 |
| VLM-RGB+Map+NavCost | 0.485000 | 0.488016 | 0.9910 | 2.942 |

Best VLM branch: **VLM-Map** (0.485000); gain over matched ScalarVisibility+Geometry: -2.50 pp.
Gain over matched MapFeature-Utility: -3.90 pp.
MapGain (VLM-Map - VLM-RGB): 1.30 pp.
RGBConditionalGain (VLM-RGB+Map - VLM-Map): 0.00 pp.
Map package availability: 100.00%.
Run status: COMPLETED.

## Decision: KILL

The current exact frame-0 pose is explicitly privileged for this viability audit; it is not a deployable claim.
If KILL, do not swap a larger VLM or fine-tune automatically; retain the matched negative result.

## Leakage flags

All formal VLM branches set uses_gt_action=false, uses_future_human_motion=false, uses_candidate_real_rgb=false, uses_candidate_real_skeleton=false and uses_candidate_recognizer_output=false.
