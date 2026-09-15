# Train-internal-Val Relative View Quality Prior — completed

Implemented and ran `activeview/scripts/experiments/run_reduced12_relative_angle_quality_prior.py` with CUDA on the frozen Yaw8Fair recognizer. The angle prior was constructed only from the raw-train-derived Yaw8 internal validation split (2,212 train source records and 245 internal-validation source records, 1,960 expanded observations; no record overlap). Moving-Val evaluation used 10,080 contexts and 68,702 Stage-A legal candidate observations; Policy Test was not read.

Key results:

- Internal angle Accuracy ranged from 0.648980 (180°) to 0.714286 (315°), a 6.531 pp gap. Accuracy ranking was 315°, 45°, 135°, 0°, 270°, 90°, 225°, 180°; margin ranking was 45°, 315°, 0°, 270°, 225°, 180°, 90°, 135°.
- Internal-to-Moving ranking Spearman was 0.142857 for Accuracy and -0.333333 for mean GT-margin: `UNSTABLE`.
- StaticPrior was 0.549901/0.571990 Accuracy/Macro-F1. RelativeAnglePrior-Acc, -F1, and -Margin reached 0.427877/0.459840, 0.430655/0.465148, and 0.413393/0.439744 respectively; the best relative prior was 11.925 pp below StaticPrior, so the preregistered decision is `KILL RELATIVE ANGLE PRIOR`.
- RGBGlobal+Angle was 0.564980/0.579979 (-0.238 pp versus RGBGlobal); GT SceneVisibility+Angle was 0.507639/0.528509 (-7.629 pp versus GT SceneVisibility). GT-action-conditioned angle prior reached 0.439683/0.469918, only +2.629 pp over unified Margin.

Artifacts are under `experiments/reduced12_eight_placement_v1/relative_angle_quality_prior/`. This is a privileged, non-deployable diagnostic because Moving-Val scoring uses GT body yaw; no recognizer or selector was trained or modified.
