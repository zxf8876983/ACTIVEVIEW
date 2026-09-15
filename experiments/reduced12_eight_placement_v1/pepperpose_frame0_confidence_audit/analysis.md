# Frame0 True-Facing + YOLO Confidence Angle Prior Audit

This is a privileged, non-deployable diagnostic. The angle prior is frozen from raw-train-derived Yaw8 internal validation; Policy Test was not read.

## Data and schema

Moving Val has 10,080 contexts and 68,702 legal candidate samples. Internal Yaw8 has 1,960 expanded observations from 245 source records.
The internal archive exposes exact `(30,17)` confidence and therefore uses `mean(confidence[0,:])`. The eight-placement policy archive exposes only `(32,)` sequence-level means; it has no future-candidate frame-0 keypoint confidence. Moving confidence landscapes are explicitly labelled proxy-only and were not used to tune the prior.

## True-facing sanity

Analytic convention: **PASS: local +Z forward; R_body_world = R_scene_yaw @ R_AMASS_root_frame0**. Old placement-yaw bins changed in 15/20 (0.750) samples. RGB debug images are unavailable because no RGB was regenerated.

## Internal frame-0 confidence landscape

| Relative angle | Count | Mean frame-0 confidence |
|---:|---:|---:|
| 0 | 245 | 0.937018 |
| 45 | 245 | 0.868478 |
| 90 | 245 | 0.744845 |
| 135 | 245 | 0.713846 |
| 180 | 245 | 0.766150 |
| 225 | 245 | 0.768666 |
| 270 | 245 | 0.776296 |
| 315 | 245 | 0.897216 |

Best internal angle=0°, worst=135°, gap=0.223172.

## Moving-Val selectors

| Method | Accuracy | Macro-F1 | Move rate |
|---|---:|---:|---:|
| Random | 0.426786 | 0.450423 | 1.000000 |
| StaticPrior | 0.549901 | 0.571990 | 1.000000 |
| GTYaw-PoseConfidencePrior | 0.437996 | 0.465250 | 1.000000 |
| RGBGlobal-Visibility | 0.567361 | 0.582567 | 1.000000 |
| Frame0SceneVisibility | 0.583929 | 0.598955 | 1.000000 |
| GT-TrueLogP Oracle | 0.760714 | 0.775961 | 1.000000 |
| AnyCorrect Coverage | 0.776190 coverage | — | — |

Internal→Moving confidence Spearman (proxy, because Moving confidence is sequence-level) = 0.809524.
Registered decision: **KILL PEPPERPOSE-STYLE ANGLE PRIOR** (GTYaw-PoseConfidencePrior Accuracy=0.437996; exact frame-0 Moving confidence is unavailable).

## Fusion

Fusion was not run because the privileged pose prior did not reach the preregistered 0.54 gate.

## Scientific answer

The true-facing angle convention is analytically consistent, but the existing policy archive cannot support the requested exact Moving frame-0 YOLO-confidence landscape: its stored confidence is a 30-frame sequence mean. Therefore this audit does not claim an exact frame-0 confidence-angle generalization result. The privileged selector number above is valid only as a true-facing prior whose internal prior is frame-0-derived; the confidence correlation gate is explicitly proxy-qualified.
No deployable yaw estimator was trained and no existing recognizer/checkpoint/data was modified.

Flags: `policy_test_used=false`, `new_rgb_generated=false`, `new_skeleton_generated=false`, `selector_trained=false`, `deployable_angle_prior=false`, `moving_frame0_confidence_exact=false`.
