# Train-Internal-Val Relative View Quality Prior

The prior was frozen from raw-train-derived Yaw8 internal validation (245 source records, 1960 expanded observations). No raw-val/Moving-Val statistic was used to construct or reorder it, and Policy Test was not read.

## Internal-val angle landscape

| Relative angle | Count | Accuracy | Macro-F1 | Mean TrueLogP | Mean GT-Margin |
|---:|---:|---:|---:|---:|---:|
| 0 | 245 | 0.685714 | 0.665183 | -1.056786 | 0.875528 |
| 45 | 245 | 0.702041 | 0.685117 | -1.050315 | 0.897095 |
| 90 | 245 | 0.673469 | 0.667589 | -1.123493 | 0.782862 |
| 135 | 245 | 0.693878 | 0.673961 | -1.093675 | 0.761117 |
| 180 | 245 | 0.648980 | 0.622271 | -1.102529 | 0.784389 |
| 225 | 245 | 0.673469 | 0.661325 | -1.052445 | 0.822635 |
| 270 | 245 | 0.681633 | 0.665616 | -1.071347 | 0.858625 |
| 315 | 245 | 0.714286 | 0.709082 | -1.069109 | 0.880304 |

Accuracy ranking (best→worst): [315, 45, 135, 0, 270, 90, 225, 180].
Macro-F1 ranking (best→worst): [315, 45, 135, 90, 270, 0, 225, 180].
GT-margin ranking (best→worst): [45, 315, 0, 270, 225, 180, 90, 135].
Internal best angle=315° and worst angle=180°; best-worst Accuracy gap=6.531pp.
Ranking correlations (Spearman): Acc↔Margin=0.595238, Acc↔F1=0.833333, F1↔Margin=0.357143.

## Moving-Val selectors

The selector rows below use GT body yaw only for this privileged angle diagnostic; terminal labels come from archived candidate ST-GCN outputs.

| Method | Accuracy | Macro-F1 | Move rate |
|---|---:|---:|---:|
| Random legal | 0.414286 | 0.436304 | 0.843254 |
| StaticPrior | 0.549901 | 0.571990 | 1.000000 |
| RelativeAnglePrior-Acc | 0.427877 | 0.459840 | 1.000000 |
| RelativeAnglePrior-F1 | 0.430655 | 0.465148 | 1.000000 |
| RelativeAnglePrior-Margin | 0.413393 | 0.439744 | 1.000000 |
| RGBGlobal-Visibility | 0.567361 | 0.582567 | 1.000000 |
| RGBGlobal+Angle | 0.564980 | 0.579979 | 1.000000 |
| Frame0SceneVisibility | 0.583929 | 0.598955 | 1.000000 |
| GT SceneVisibility+Angle | 0.507639 | 0.528509 | 1.000000 |
| GTActionRelativeAnglePrior | 0.439683 | 0.469918 | 1.000000 |
| GT-TrueLogP Oracle | 0.760714 | 0.775961 | 1.000000 |

The best relative selector by Moving-Val Accuracy is RelativeAnglePrior-F1 (0.430655/0.465148), -11.925pp Accuracy versus StaticPrior (0.549901/0.571990).
Decision by the preregistered threshold is **KILL RELATIVE ANGLE PRIOR**; this is a report-only comparison, not Moving-Val prior tuning.
Static/relative selected-view agreement=0.275794; Static-correct/relative-wrong=2089, Static-wrong/relative-correct=713, both-wrong=3824; pair AnyCorrect=0.620635.

## Fusion and action-conditioned diagnostics

RGBGlobal-Visibility=0.567361/0.582567; RGBGlobal+Angle uses train-record holdout lambda=0.25 and gives 0.564980/0.579979 (-0.238pp Accuracy).
Frame0SceneVisibility=0.583929/0.598955; GT SceneVisibility+Angle uses lambda=0.25 and gives 0.507639/0.528509 (-7.629pp Accuracy).
GTActionRelativeAnglePrior=0.439683/0.469918; gain over unified Margin prior=+2.629pp Accuracy.
Per-action internal preferred-angle examples (best angle / worst angle / gap):
- walk: 270° / 90° / 13.333pp
- sit: 315° / 180° / 10.000pp
- stand up: 0° / 270° / 6.667pp
- bend: 315° / 225° / 23.333pp
- crawl: 90° / 0° / 14.286pp
- stumble: 315° / 45° / 33.333pp
- clap: 90° / 45° / 20.000pp
- throw: 45° / 135° / 23.333pp
- kick: 135° / 90° / 13.333pp
- knock: 45° / 0° / 14.286pp
- punch: 45° / 0° / 14.286pp
- touching face: 90° / 180° / 33.333pp

## High-occlusion diagnostic

Fixed lower-tertile current-slot visibility subset: n=3271 (no prior tuning on this subset).

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| StaticPrior | 0.481810 | 0.512912 |
| RelativeAnglePrior-Margin | 0.308468 | 0.340664 |
| RGBGlobal-Visibility | 0.515439 | 0.534703 |
| RGBGlobal+Angle | 0.514216 | 0.533764 |
| Frame0SceneVisibility | 0.520024 | 0.539481 |
| GTActionRelativeAnglePrior | 0.355854 | 0.388656 |
| GT-TrueLogP Oracle | 0.664934 | 0.689528 |

## Required scientific answers

1. Frozen Yaw8Fair shows a modest internal angle landscape (best 315° vs worst 180°, 6.531pp), but its internal-to-Moving ranking correlations (0.143 Accuracy, -0.333 margin) are UNSTABLE rather than stable.
2. The relative prior reaches only 0.430655 Accuracy / 0.465148 Macro-F1, versus StaticPrior 0.549901/0.571990; it is 11.925pp below StaticPrior and therefore not stronger than the absolute prior.
3. RGB angle fusion changes RGBGlobal by -0.238pp, while GT SceneVisibility fusion changes its baseline by -7.629pp; neither supplies a positive gain here.
4. The GT-action-conditioned ceiling is 0.439683 Accuracy, only +2.629pp above the unified margin prior. This is evidence against a large, deployable soft action-belief angle prior at this stage.
5. Final decision: **KILL RELATIVE ANGLE PRIOR**. Body yaw and future candidate metadata make this a privileged, non-deployable diagnostic; no selector or recognizer was trained.

Flags: `policy_test_used=false`, `raw_val_used_to_construct_angle_prior=false`, `moving_val_used_for_angle_prior_selection=false`, `recognizer_modified=false`, `selector_trained=false`, `deployable_angle_prior=false`.
