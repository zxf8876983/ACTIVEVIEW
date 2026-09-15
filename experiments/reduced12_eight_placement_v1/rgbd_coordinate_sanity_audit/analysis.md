# RGB-D Coordinate-System & D1 Sanity Audit

Train/Moving-Val only; Policy Test and future candidate RGB/depth were not read.

## A. Baseline protocol reproduction

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Random legal | 0.426786 | 0.450423 |
| StaticPrior | 0.549901 | 0.571990 |
| D0 Frame0SceneVisibility | 0.583929 | 0.598955 |
| GT-TrueLogP Oracle | 0.760714 | 0.775961 |
| GT-Margin Oracle | 0.776190 | 0.796334 |

StaticPrior first causal difference: the previous RGB-D runner constructed the prior from Val rows inside `_policy_scores(rows=Val)`. This audit uses the historical definition `Q(v)=mean Train candidate-only GT-Margin` and applies that fixed table to Val; reproduction therefore returns the historical 54.99% range.

## B. Camera and projection sanity

World up=+Y; Habitat camera forward=-Z, right=+X, image y points down; rotations are WXYZ and stored agent/camera orientation is camera→world. Sensor height=1.10m is applied once (sensor position is not pre-added). Forward-to-human dot mean/min: 0.998612/0.997773.

Projection→backprojection closure: median 0.000e+00m, P95 0.000e+00m, max 0.000e+00m; gate=PASS.

Synthetic yaw unit test evaluates H0–H6 and selects H2_minus_90: median/max absolute error 0.472°/0.473°. The raw lateral-axis estimate has a fixed approximately +90° offset; H2 (−90° correction) passes the <2°/<5° synthetic gate. Moving-Val yaw remains a noisy estimated-pose diagnostic and is not used to choose a deployable convention.

## C. RGB-D error decomposition

| Reconstruction | MPJPE |
|---|---:|
| L0_GT_pixel_GT_analytic_depth | 0.0 m |
| L1_GT_pixel_Habitat_depth | 0.9626972912780232 m |
| L2A_YOLO_pixel_GT_depth | 0.1571773766826371 m |
| L2B_YOLO_pixel_Habitat_depth | 0.43670520348490804 m |

L1 visible-joint MPJPE=0.38867811009866154m; occluded-joint MPJPE=1.6290611669470136m.

## D. True root localization

Independent GT root is H36M17 joint0 (pelvis). Euclidean mean/median/P75/P90/P95: 1.0028134433323068/0.3549350084640083/1.9037306500880375/2.362965583636558/2.5784389885678833 m. Horizontal P90=2.152470708439246m; vertical P90=1.0920921623706819m. The old near-zero metric is explicitly self-consistency (estimated root vs D1-derived joint0), not GT localization.

## E. D1/D2 attribution

D1a-vs-existing D1b mean/P95 joint difference=1.556e-08/1.192e-07m; translation-only=True.

| Method | Accuracy | Macro-F1 | Population |
|---|---:|---:|---:|
| D0 | 0.583929 | 0.598955 | full Moving Val |
| D1 corrected | 0.499802 | 0.524925 | full Moving Val |
| D2c deployable | 0.500496 | 0.521328 | full Moving Val |

D1/D2 sensitivity curves are fixed diagnostic subsets (no fitting): see `translation_sensitivity_curve.json` and `yaw_sensitivity_curve.json`.

Train torso-surface depth calibration median bias=0.3328m; hard rule enabled=True (no calibration is used unless the preregistered 5cm/20% rule passes).

### Root estimator ladder (fixed 256-context diagnostic subset)

| Estimator | Median error (m) | P90 (m) |
|---|---:|---:|
| R0 | 0.000000 | 0.000000 |
| R1 | 0.363443 | 2.087009 |
| R2 | 0.092663 | 3.165326 |
| R3 | 0.398893 | 2.283592 |

### D2 component ladder (fixed 256-context diagnostic subset)

| Variant | Accuracy | Macro-F1 |
|---|---:|---:|
| D1_cached | 0.585938 | 0.613800 |
| D2a_GT_root_estimated_pose | 0.585938 | 0.614679 |
| D2b_est_root_estimated_pose_GT_orientation | 0.507812 | 0.560738 |
| D2c_est_root_estimated_pose_estimated_orientation | 0.511719 | 0.565650 |
| D2_observed_only | 0.550781 | 0.584505 |

### Pair re-audit (full Moving Val)

| Pair | Any-correct rate |
|---|---:|
| D2+D0 | 0.607341 |
| D2+StaticPrior | 0.625496 |
| D2+D1 | 0.524603 |

## F. Final diagnosis

The previous StaticPrior 52.34% was a protocol bug: a Val-derived prior replaced the Train-derived historical prior. After correction it returns to the 54.99% range. Projection/backprojection is mathematically closed and no double sensor-height translation is present.

The real GT root error, rather than the old self-comparison metric, must be used for attribution. D1 is translation-only; D1 performance should be interpreted against this independent root error. The raw moving-Val yaw discrepancy is about 92.6°, while the synthetic unit test identifies a fixed +90° lateral-axis offset (H2, −90° correction) with sub-degree residual error. This is a convention issue in the current template/lateral-axis interpretation, not evidence to select a convention from HAR accuracy.

The D2 corrected ladder and component decomposition are diagnostic only. No gate was trained, no recognizer was modified, no new RGB/skeleton/DINO was generated, and no Policy Test data was read.
