# Yaw8 Encoder + Policy-Balanced Shared Head Fairness Audit

Purpose: remove the classifier-head training-distribution mismatch from the previous Old-vs-Yaw8 policy landscape comparison.

Protocol: both ST-GCN encoders are frozen; each head is Linear(256,256) → GELU → Linear(256,12), trained on identical Policy-Train record-balanced samples (16 observations per record per epoch). A deterministic 10% Policy-Train record holdout selects checkpoints. Moving Val is evaluation-only and Policy Test was not read.

## Historical Old shared-head reproduction

Stay=0.302579, legal micro=0.375418, Random=0.365079; the historical reproduction gate is within ±0.2pp.

## Main matched-head Moving-Val results

| Method | OldFair Acc/F1 | Yaw8Fair Acc/F1 | Yaw8Native Acc/F1 |
|---|---:|---:|---:|
| Stay/current | 0.290575/0.290338 | 0.350496/0.364253 | 0.316865/0.330391 |
| Random legal | 0.353770/0.359653 | 0.414286/0.436304 | 0.383433/0.410639 |
| StaticPrior | 0.472718/0.486772 | 0.548611/0.571334 | 0.530556/0.545988 |
| GT-TrueLogP Oracle | 0.736607/0.744334 | 0.774603/0.787987 | 0.773611/0.790086 |
| GT-Margin Oracle | 0.761409/0.777916 | 0.791369/0.811436 | 0.777976/0.794831 |
| Legal candidate micro | 0.361838/0.370739 | 0.429245/0.453313 | 0.399042/0.427756 |

Candidate-only oracle (primary legal action scope): OldFair GT-TrueLogP=0.717659, GT-Margin=0.740476, AnyCorrect=0.740476; Yaw8Fair GT-TrueLogP=0.760714, GT-Margin=0.776190, AnyCorrect=0.776190.

## Matched-old-head sanity check

Compared with the historical Old shared head, the newly retrained OldFair head changes Stay accuracy by -1.200pp, legal-micro accuracy by -1.358pp, and Random accuracy by -1.131pp. The >1pp shift indicates head-retraining sensitivity; OldFair↔Yaw8Fair remains a controlled paired comparison, but promotion should be interpreted as a matched-protocol signal rather than an unconditional historical replication.

## Fair encoder and policy-head gains

Yaw8Fair − OldFair: Stay +5.992pp, legal micro +6.741pp, Random +6.052pp, StaticPrior +7.589pp.
Yaw8 shared head − Yaw8 native FC: legal micro +3.020pp, Random +3.085pp.

## Relative-yaw audit

OldFair best-worst candidate accuracy gap=5.118pp; Yaw8Fair gap=10.274pp. These policy yaw bins are confounded by scene geometry/occlusion and are not a pure yaw-invariance test.

## Per-class observations

Matched shared-head Random changes are reported in `per_class_metrics.json`; the key native-FC declines should not be attributed to the encoder until this matched comparison is considered.

## Historical controlled clean-yaw evidence (not rerun)

Old Yaw8 Val = 58.3163/54.5710; Yaw8 model = 72.5510/72.7956; MeanYawGain = +14.2347pp; WorstYawGain = +20.4082pp. This controlled clean-yaw result is separate from scene-conditioned policy relative-angle behavior.

## Decisions

**Promotion: STRONG PROMOTE YAW8 ENCODER.** This decision uses only matched-head legal/Random gains, Macro-F1 behavior, and the largest key-class Random decline (7.506pp). Because the matched Old head differs from the historical head by more than 1pp, this promotion is a controlled paired-encoder result and should not be read as an unconditional absolute claim.
**NBV viability: SUBSTANTIAL NBV HEADROOM REMAINS.** Yaw8Fair StaticPrior to GT-TrueLogP gap=22.599pp; AnyCorrect to StaticPrior gap=24.276pp.

A. Under fair Policy-balanced head adaptation, Yaw8 is compared to Old only through the frozen encoder representation; the head architecture, sampler, optimizer, seed and evaluation action set are identical.

B. If a future run promotes Yaw8, freeze the encoder and matched head, rebuild Policy-Train utility targets, then retrain NBV. No selector was trained in this audit.

```text
policy_test_used=false
moving_val_used_for_head_training_or_selection=false
new_rgb_generated=false
new_skeleton_generated=false
yolo_rerun=false
videopose3d_rerun=false
old_encoder_modified=false
yaw8_encoder_modified=false
```
