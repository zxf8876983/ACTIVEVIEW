# True-Facing ST-GCN Angle Prior Re-Audit

This is a read-only Train-internal-Val and Moving-Val diagnostic. The body-facing angle composes the AMASS frame-0 root rotation with scene yaw, then bins camera world azimuth relative to that facing. Policy Test was not read and no model/data was generated.

Internal bin alignment changed 1440/1960 (0.734694); Moving legal candidates changed 57609/68702 (0.838535).

## Corrected internal angle landscape

| Angle | Count | Accuracy | Macro-F1 | Mean GT-TrueLogP | Mean GT-Margin |
|---:|---:|---:|---:|---:|---:|
| 0 | 245 | 0.693878 | 0.687464 | -1.061198 | 0.838470 |
| 45 | 245 | 0.693878 | 0.670383 | -1.091558 | 0.841644 |
| 90 | 245 | 0.677551 | 0.637504 | -1.122120 | 0.790725 |
| 135 | 245 | 0.657143 | 0.633744 | -1.148443 | 0.722954 |
| 180 | 245 | 0.697959 | 0.684123 | -1.070219 | 0.832321 |
| 225 | 245 | 0.706122 | 0.698565 | -1.029664 | 0.923692 |
| 270 | 245 | 0.653061 | 0.656628 | -1.048955 | 0.845999 |
| 315 | 245 | 0.693878 | 0.688020 | -1.047543 | 0.866750 |

Internal corrected best/worst Accuracy bins: 225°/270°.
Moving corrected best/worst Accuracy bins: 225°/45°.
Train→Moving Spearman: Q_acc vs accuracy=0.238095; Q_margin vs mean margin=0.547619. Historical old values were 0.142857 and -0.333333.

## Moving selector metrics

| Selector | Accuracy | Macro-F1 | Move rate |
|---|---:|---:|---:|
| Random | 0.426786 | 0.450423 | 1.000000 |
| StaticPrior | 0.549901 | 0.571990 | 1.000000 |
| Old RelativeAnglePrior-Acc | 0.427877 | 0.459840 | 1.000000 |
| Old RelativeAnglePrior-Margin | 0.413393 | 0.439744 | 1.000000 |
| TrueFacingAnglePrior-Acc | 0.450099 | 0.477597 | 1.000000 |
| TrueFacingAnglePrior-Margin | 0.451587 | 0.483006 | 1.000000 |
| RGBGlobal-Visibility | 0.567361 | 0.582567 | 1.000000 |
| Frame0SceneVisibility | 0.583929 | 0.598955 | 1.000000 |
| GT-TrueLogP Oracle | 0.760714 | 0.775961 | 1.000000 |
| AnyCorrect Coverage | 0.776190 coverage | — | — |

Decision: **KILL ST-GCN RELATIVE-ANGLE PRIOR**.

### Interpretation

A. The corrected prior changes the angle assignment for the fractions above; the measured selector accuracy is 0.451587 for TrueFacingAnglePrior-Margin, versus StaticPrior 0.549901.
B. Corrected internal and Moving angle preference agree only to the extent captured by the two reported Spearman values; the old placement-yaw table is retained for direct comparison.
C. If the corrected prior remains weak or unstable, the next diagnostic should be Clean-to-Observed ST-GCN feature degradation rather than another angle prior. No fusion or new model was attempted.
