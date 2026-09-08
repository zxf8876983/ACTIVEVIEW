# Reduced12 H1 discriminative objective batch

Val moving contexts: 10080
Train Stage-C rows: 46324

## H1 ceiling and objective results

| H1 | Accuracy | Macro-F1 | Move rate | Selected candidate correct |
|---|---:|---:|---:|---:|
| S0-only | 0.254266 | 0.235500 | 0.000000 | 0.000000 |
| FrozenStageCv0 | 0.454266 | 0.444782 | 1.000000 | 0.454266 |
| Baseline reproduction | 0.454266 | 0.444782 | 1.000000 | 0.454266 |
| AnyCorrect Oracle | 0.728274 | 0.729637 | 0.474008 | 1.000000 |
| GTTrueLogP Oracle | 0.727976 | 0.721495 | 0.886706 | 0.736406 |
| GTMargin Oracle | 0.728274 | 0.722059 | 0.887103 | 0.736524 |
| correctness_bce | 0.275893 | 0.265793 | 0.246726 | 0.611982 |
| gt_logp | 0.254266 | 0.235500 | 0.000198 | 1.000000 |
| gt_margin | 0.269544 | 0.260030 | 0.178373 | 0.635150 |
| multi_positive | 0.450794 | 0.440725 | 0.881548 | 0.478506 |
| hybrid | 0.263194 | 0.248588 | 0.121528 | 0.714286 |

## Per-class recall/F1

| Action | Frozen Recall/F1 | Correctness-BCE Recall/F1 | GTTrueLogP Recall/F1 | GTMargin Recall/F1 | Multi-positive Recall/F1 | Hybrid Recall/F1 |
|---|---:|---:|---:|---:|---:|---:|
| walk | 0.5929/0.4006 | 0.5070/0.2609 | 0.5042/0.2622 | 0.5042/0.2615 | 0.5705/0.4090 | 0.5098/0.2636 |
| sit | 0.7201/0.7215 | 0.3740/0.4203 | 0.3346/0.3620 | 0.3397/0.3809 | 0.7341/0.7397 | 0.3448/0.3730 |
| stand up | 0.6174/0.6800 | 0.3616/0.4400 | 0.2999/0.3713 | 0.3286/0.4024 | 0.6174/0.6561 | 0.3385/0.4129 |
| bend | 0.2826/0.3199 | 0.3093/0.2003 | 0.2915/0.1881 | 0.3113/0.1984 | 0.3083/0.2959 | 0.3043/0.1955 |
| crawl | 0.7354/0.7673 | 0.3677/0.4816 | 0.2556/0.3540 | 0.3744/0.4848 | 0.7152/0.7559 | 0.3117/0.4174 |
| stumble | 0.1602/0.2373 | 0.0448/0.0820 | 0.0224/0.0429 | 0.0400/0.0735 | 0.1449/0.2234 | 0.0318/0.0597 |
| clap | 0.3930/0.4202 | 0.1111/0.1600 | 0.1152/0.1628 | 0.1235/0.1727 | 0.3765/0.4071 | 0.1152/0.1607 |
| throw | 0.5717/0.5009 | 0.2549/0.3189 | 0.2405/0.3076 | 0.2590/0.3228 | 0.5521/0.4853 | 0.2405/0.3078 |
| kick | 0.7543/0.5768 | 0.4601/0.4033 | 0.4523/0.3710 | 0.4488/0.4016 | 0.7335/0.5860 | 0.4453/0.3853 |
| knock | 0.1625/0.2462 | 0.0632/0.1057 | 0.0632/0.1061 | 0.0655/0.1094 | 0.1535/0.2321 | 0.0655/0.1094 |
| punch | 0.1749/0.2102 | 0.0806/0.1113 | 0.0754/0.1070 | 0.0827/0.1132 | 0.1749/0.2148 | 0.0775/0.1080 |
| touching face | 0.2141/0.2564 | 0.1643/0.2052 | 0.1494/0.1911 | 0.1594/0.1993 | 0.2430/0.2834 | 0.1494/0.1898 |

Best deployable objective on Val Moving: `multi_positive` (0.450794 accuracy, 0.440725 Macro-F1; ΔAccuracy vs Frozen -0.003472).
The AnyCorrect Oracle ceiling is 0.728274, leaving 0.274008 accuracy points above FrozenStageCv0.
The baseline reproduction passed the pre-registered ±0.5pp gate; no downstream branch was stopped.

## Scientific reading

None of the discriminative objectives exceeded the frozen Stage-C-v0 H1 baseline on Val Moving. Multi-positive ranking was closest (within 0.35pp accuracy), but did not improve Macro-F1. The large AnyCorrect Oracle gap indicates that first-step view opportunity remains, while these scalar objectives did not learn it better from the fixed candidate geometry. Therefore this batch does not justify connecting a new H1 objective to WM-E/H2; preserve FrozenStageCv0 and treat objective alignment as unresolved.

No Test files were read; no skeleton/perception/RGB/DINO data were regenerated.
