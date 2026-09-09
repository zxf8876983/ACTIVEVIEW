# Reduced12 H1 visual-context representation batch

Train/Val only. No policy Test was read and no skeleton, RGB or DINO data was regenerated.

Visual DINO coverage: Train 30580/46324 Stage-C rows and moving Val 10080/15540 rows. The existing cache does not cover non-moving Full-Val contexts; Full-Val visual metrics are therefore reported as NOT_AVAILABLE rather than using a silent fallback.

## Moving Val

| H1 | Accuracy | Macro-F1 | Move rate | Rescue | Harm | GTMargin regret |
|---|---:|---:|---:|---:|---:|---:|
| FrozenStageCv0 | 0.454266 | 0.444782 | 1.000000 | 0.380870 | 0.330472 | 2.598949 |
| GTMargin Listwise | 0.458730 | 0.446342 | 0.972024 | 0.374351 | 0.293796 | 2.596994 |
| AnyCorrect Oracle | 0.728274 | 0.729637 | 0.474008 | 0.635626 | 0.000000 | 1.196742 |
| dino_mean | 0.466567 | 0.459051 | 0.918452 | 0.384861 | 0.293796 | 2.570037 |
| dino_meanmax | 0.466667 | 0.458637 | 0.920734 | 0.383797 | 0.290285 | 2.538825 |
| dino_spatial | 0.468552 | 0.459182 | 0.906944 | 0.386457 | 0.290675 | 2.600325 |
| candidate_conditioned_spatial | 0.471528 | 0.463723 | 0.922520 | 0.386457 | 0.278970 | 2.498159 |

## Full Val reference

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| FrozenStageCv0 | 0.460167 | 0.450040 |
| GTMargin Listwise | 0.464865 | 0.452845 |
| AnyCorrect Oracle | 0.731338 | 0.731446 |

Full-Val visual branch metrics are intentionally not reported because the existing visited s0 DINO cache covers only the moving-Val contexts; generating missing DINO is prohibited in this experiment.

Best visual branch: **candidate_conditioned_spatial**, Moving Accuracy gain over FrozenStageCv0: **+1.726 pp**.

## Scientific decision

All visual-context branches stay below +2 pp over FrozenStageCv0; this batch does not establish that a single visited s0 visual context is sufficient to predict candidate recognition outcomes.
The fixed Stay-aware GTMargin listwise objective was reused without loss changes. No branch was connected to H2.

## Focus classes

Per-class Recall/F1 for bend, stumble, knock and touching face are in `per_class_metrics.json`.

test_used=false; future_candidate_dino_used=false; skeleton/rgb/dino regenerated=false.
