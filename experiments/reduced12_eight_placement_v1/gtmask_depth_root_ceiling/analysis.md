# GT Human Mask + Depth Root Recovery Ceiling Audit

Policy Train was used only for the pelvis-height prior and one global radial calibration; Moving Val was evaluated with current frame 0 only. Policy Test, candidate RGB/depth, future frames, YOLO, VideoPose3D and DINO were not used.

## Baselines and D1

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| StaticPrior | 0.549901 | 0.571990 |
| D0 GT-human-state SceneVisibility | 0.583929 | 0.598955 |
| D1 old joint-depth root | 0.499802 | 0.524925 |
| D1 GTMask RawRoot | 0.553671 | 0.572959 |
| D1 GTMask CalibratedRoot | 0.556548 | 0.576580 |
| Oracle GT-TrueLogP | 0.760714 | 0.775961 |

## Root localization

GTMask RawRoot euclidean mean/median/P75/P90/P95: 1.435254/0.394503/2.795847/3.193952/5.322804 m.
GTMask CalibratedRoot euclidean mean/median/P75/P90/P95: 1.345525/0.269271/2.784600/3.172190/5.314395 m.
Existing old joint-depth root reference euclidean median/P90: 0.354935/2.362966 m.
Valid GTMask point-cloud roots: Raw 7051/10080; Calibrated 7051/10080.
Train-derived pelvis height prior: 0.826920 m; radial offset b: 0.272175 m.

The mask is Habitat semantic OBJECT_ID for the articulated humanoid and the point cloud uses all finite human-mask depth pixels. Raw depth/masks are transient and no point cloud is serialized.

## Required interpretation

D0 accuracy is 0.583929; old joint-depth D1 is 0.499802; GTMask RawRoot D1 is 0.553671; GTMask CalibratedRoot D1 is 0.556548.
D0 versus best GTMask D1 drop: 2.738 pp. Selected-view agreement D0→Raw/Calibrated D1: 0.884325/0.892956.

1. The comparison is against the independent GT H36M17 pelvis, not estimated-vs-estimated self-consistency.
2. A global Train-only radial offset is reported; no scene-, action-, distance-bin- or Val-derived calibration is used.
3. If GTMask D1 remains far below D0, perfect human segmentation does not by itself make current depth sufficient for known-map NBV; the simple RGB-D root route should be stopped rather than expanded with more point-cloud heuristics.
4. Localization classification: **KILL SIMPLE RGB-D ROOT LOCALIZATION**; route classification: **KILL SIMPLE RGB-D ROOT LOCALIZATION**.

## Direct answers

1. Root median/P90 changes from the old 0.354935/2.362966 m to GTMask calibrated 0.269271/3.172190 m (raw 0.394503/3.193952 m).
2. The global Train-only radial calibration is helpful for the median and horizontal error (calibrated radial offset b=0.272175 m), but it does not materially reduce the long tail.
3. D1 recovers from 0.499802 to 0.556548 Accuracy; it remains 2.738 pp below D0.
4. The improvement over joint-depth D1 supports pixel-depth association as a major contributor to the earlier failure, but the remaining D0 gap shows it is not the only bottleneck.
5. With a perfect human mask, depth reaches 55.655% D1 Accuracy rather than the 56–58% D0 range; this is insufficient to claim that depth alone supports the known-map NBV route.
6. A deployable human-segmentation follow-up is not justified by this ceiling audit; stop the simple RGB-D localization route before adding segmentation engineering.

## High occlusion

The fixed strict lower tertile of current-slot D0 visibility contains 3271 contexts. See `high_occlusion_metrics.json` for StaticPrior, D0, Raw D1, Calibrated D1 and Oracle.

No next-stage deployable mask work was started automatically.
