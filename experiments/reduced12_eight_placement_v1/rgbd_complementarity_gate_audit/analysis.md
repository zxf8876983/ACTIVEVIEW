# Selector Complementarity + RGB-D Deployable NBV Audit

Protocol: strict current Frame0 -> one Stage-A candidate-only action -> real archived O1 Yaw8Fair HAR.
Policy Test used: false. Moving Val was not used for training or threshold selection.

## Phase A: selector metrics (Moving Val)

| Method | Accuracy | Macro-F1 | Move rate |
|---|---:|---:|---:|
| Frame0SceneVisibility | 0.583929 | 0.598955 | 1.000000 |
| OldTargetSelector→Yaw8Fair | 0.572222 | 0.587019 | 1.000000 |
| RGBGlobal-Visibility | 0.567361 | 0.582567 | 1.000000 |
| StaticViewPrior | 0.549901 | 0.571990 | 1.000000 |
| G_full λ=.5 | 0.583532 | 0.602979 | 1.000000 |
| Random legal | 0.426786 | 0.450423 | 1.000000 |

Pair AnyCorrect rates: A+B=0.620933, A+C=0.633631, A+D=0.640377, A+E=0.662004, B+C=0.617262, B+D=0.630159, C+D=0.643254
Best pair containing A: 0.662004; gate decision: GATING AUDIT ALLOWED.

## Phase B: RGB-D capability

Current depth cache present: False; raw frame-0 YOLO cache present: False.
Habitat SensorType.DEPTH probe: PASS; the probe rendered 50 Train + 50 Moving-Val current-frame depth images transiently, but no raw depth cache was retained. Raw frame-0 YOLO remains absent, so D1/D2/D3 and gates are N/A rather than fabricated.
Current RGB archives expose image size, camera positions/rotations and yaw metadata, but not explicit depth calibration or per-joint frame-0 confidence.

## Scientific conclusion

Pair complementarity clears the preregistered threshold; only a Train-internal gate selection would be admissible after current-frame RGB-D artifacts are acquired and aligned.

Explicit limitations: A uses GT frame-0 human state and is privileged; candidate skeleton/logits are used only after selection for terminal evaluation. No claim of deployability is made for D0/A.
