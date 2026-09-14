# Yaw8Fair Strict-Frame0 NBV Full Re-baseline — completed

The strict candidate-only Frame0 protocol was re-evaluated with the frozen
Yaw8 ST-GCN encoder and matched Policy-balanced shared head. Policy Train used
46,324 contexts across 313 records for selector fitting; Moving Val used
10,080 contexts across 105 records for evaluation. No Policy Test, new
perception data or recognizer changes were used. The Stage-A legal candidate
pool had mean/min/max size 6.8157/2/21.

Moving-Val Accuracy/Macro-F1: Random legal 0.426786/0.450423,
StaticViewPrior 0.549901/0.571990, Frame0SceneVisibility
0.583929/0.598955, best learned Prior+RGBResidual λ=0.5
0.550794/0.573851, GT-TrueLogP Oracle 0.760714/0.775961 and GT-Margin /
AnyCorrect 0.776190/0.796334 and 0.776190 coverage. Frame0SceneVisibility is
the best strict method (+3.403pp over StaticViewPrior); learned utility
residual gain is only +0.089pp. Decision: keep the instance-conditioned
Frame0 NBV baseline, but do not automatically start another method family.

Artifacts: `experiments/reduced12_eight_placement_v1/yaw8_strict_frame0_full_rebaseline/`.
Runtime checkpoints remain external under `ACTIVEVIEW_DATA_ROOT`.
