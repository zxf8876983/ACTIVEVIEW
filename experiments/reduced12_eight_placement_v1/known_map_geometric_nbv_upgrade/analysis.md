# Known-Map Geometric NBV Upgrade Suite

Protocol: Frame0 legal information → exactly one Stage-A legal candidate → selected real O1 alone → frozen Yaw8Fair HAR.
Moving Val only (10,080 contexts); Policy Test was never read. Map quantities are privileged GT-frame-0 geometry diagnostics.

## Main privileged results

| Method | Accuracy | Macro-F1 | Gain vs StaticPrior | Gain vs Frame0Visibility |
|---|---:|---:|---:|---:|
| Random legal | 0.426786 | 0.450423 | -12.312pp | -15.714pp |
| StaticViewPrior | 0.549901 | 0.571990 | +0.000pp | -3.403pp |
| Frame0SceneVisibility | 0.583929 | 0.598955 | +3.403pp | +0.000pp |
| Visibility+Prior λ=0.25 | 0.582143 | 0.597840 | +3.224pp | -0.179pp |
| Visibility+Prior λ=0.5 | 0.582044 | 0.597594 | +3.214pp | -0.188pp |
| Visibility+Prior λ=1.0 | 0.581250 | 0.597213 | +3.135pp | -0.268pp |
| VisibilityPrior-RankSum | 0.581151 | 0.596725 | +3.125pp | -0.278pp |
| DenseVisibility | 0.583631 | 0.598359 | +3.373pp | -0.030pp |
| Completeness | 0.541468 | 0.563937 | -0.843pp | -4.246pp |
| ScaleOnly | 0.448115 | 0.480918 | -10.179pp | -13.581pp |
| DOQ-Equal | 0.557341 | 0.577746 | +0.744pp | -2.659pp |
| G_geo | 0.557341 | 0.577746 | +0.744pp | -2.659pp |
| G_full λ=0.25 | 0.573909 | 0.593563 | +2.401pp | -1.002pp |
| G_full λ=0.5 | 0.583532 | 0.602979 | +3.363pp | -0.040pp |
| GT-TrueLogP Oracle | 0.760714 | 0.775961 | +21.081pp | +17.679pp |
| GT-Margin Oracle | 0.776190 | 0.796334 | +22.629pp | +19.226pp |

## Required answers

1. Visibility+prior is evaluated at fixed λ=.25/.5/1 and rank-sum; the best gain over Frame0Visibility is -0.179pp.
2. Dense visibility versus 17-joint visibility: -0.030pp; completeness-only: -4.246pp; scale-only: -13.581pp.
3. Best explicit geometric method is **DenseVisibility**, Accuracy/F1=0.583631/0.598359; gain vs StaticPrior=+3.373pp and vs Frame0Visibility=-0.030pp.
4. Pre-registered decision: **STOP GEOMETRIC SCORE EXPANSION**. The 60% milestone is not crossed.
5. The dense completeness value is explicitly an H36M17 in-FOV proxy because the existing cache persisted dense ray visibility but not dense projected-point counts. Uncertainty weighting is SKIPPED because archives contain only a 30-frame viewpoint scalar confidence, not current frame-0 per-joint confidence; no joint confidence was fabricated.
6. E1/E2 estimated-state scores are N/A: the current pipeline has no legal metric global human localization/world-space estimated pose artifact. This is a state-estimation capability gap, not an imputed score.
7. High-occlusion and per-class tables are saved in high_occlusion_metrics.json and per_class_metrics.json. Correlations are saved in correlation_diagnostics.json.

## Boundary and leakage

No candidate RGB, candidate skeleton, candidate ST-GCN output, future frame, GT action, or Policy Test artifact entered a selector score. Selected archived skeletons are read only for terminal evaluation.

```text
policy_test_used=false
new_model_training=false (Train-only prior/scale statistics; optional DOQ calibration is skipped unless its pre-registered gate is met)
known_static_map_used=true
gt_frame0_human_state_used=true (privileged track)
deployable=false
```
