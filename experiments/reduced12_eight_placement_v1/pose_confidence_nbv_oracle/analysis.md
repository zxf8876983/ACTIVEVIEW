# Reduced12 Pose Observation Confidence NBV Oracle

Val Moving contexts only. Pose Observation Confidence is a viewpoint-level scalar (the archived mean YOLO keypoint confidence over the 30-frame sequence). It is used only as a privileged future-candidate diagnostic; no model was trained and no Test artifact was read.

## Metrics

| Method | Accuracy | Macro-F1 | ΔAcc vs Frozen | ΔF1 vs Frozen |
|---|---:|---:|---:|---:|
| S0-only | 0.254266 | 0.235500 | -20.000 | -20.928 |
| FrozenStageCv0 | 0.454266 | 0.444782 | +0.000 | +0.000 |
| Random | 0.322619 | 0.316416 | -13.165 | -12.837 |
| MaxPoseConfidence | 0.455754 | 0.438806 | +0.149 | -0.598 |
| AnyCorrect Oracle | 0.728274 | 0.729637 | +27.401 | +28.486 |

## Confidence audit

- Per-context Spearman(confidence, GT-class true logp): mean **0.220495**, median **0.323359**.
- Mean confidence for ST-GCN-correct candidates: **0.693494**.
- Mean confidence for ST-GCN-wrong candidates: **0.472666**.
- Correct-minus-wrong mean confidence: **+0.220827**.

## Focus classes (MaxPoseConfidence Recall/F1)

| bend | stumble | knock | touching face |
|---:|---:|---:|---:|
| 0.330/0.334 | 0.203/0.262 | 0.106/0.170 | 0.201/0.233 |

## Scientific interpretation

MaxPoseConfidence changes Moving Accuracy by **+0.149 pp** and Macro-F1 by **-0.598 pp** relative to FrozenStageCv0.
The confidence cue shows a positive candidate-quality association in this audit; the next authorized direction would be to predict future confidence from current observation, scene model and candidate geometry.
MaxPoseConfidence is privileged and deployable=false because it reads future candidate confidence. It is not connected to the policy.

test_used=false; future_candidate_confidence_used_for_target_or_oracle_only=true; no RGB/skeleton/DINO data was regenerated.
