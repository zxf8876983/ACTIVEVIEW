# Reduced12 Stay-aware H1 objective batch

Val Moving contexts: 10080

| H1 | Moving Acc | Moving Macro-F1 | Move rate | ΔAcc vs Frozen |
|---|---:|---:|---:|---:|
| S0-only | 0.254266 | 0.235500 | 0.000000 | -0.200000 |
| FrozenStageCv0 | 0.454266 | 0.444782 | 1.000000 | +0.000000 |
| Baseline reproduction | 0.454266 | 0.444782 | 1.000000 | +0.000000 |
| delta_logp | 0.445139 | 0.432075 | 0.751687 | -0.009127 |
| delta_margin | 0.453770 | 0.441772 | 0.864881 | -0.000496 |
| stay_multi_positive | 0.267857 | 0.253249 | 0.033433 | -0.186409 |
| logp_listwise | 0.452778 | 0.440213 | 0.925397 | -0.001488 |
| margin_listwise | 0.458730 | 0.446342 | 0.972024 | +0.004464 |
| hybrid | 0.314782 | 0.306061 | 0.181548 | -0.139484 |
| AnyCorrect Oracle | 0.728274 | 0.729637 | 0.474008 | +0.274008 |
| GTTrueLogP Oracle | 0.727976 | 0.721495 | 0.886310 | +0.273710 |

## Required diagnostics

Every branch includes stay/move rate, s0-error correction, s0-correct harm, selected-action positive rate, oracle-action overlap, and GTLogP regret in `result.json`.

## Per-class Recall/F1

| Action | Frozen | DeltaLogP | DeltaMargin | Stay-Multi | LogP-Listwise | Margin-Listwise | Hybrid |
|---|---:|---:|---:|---:|---:|---:|---:|
| walk | 0.5929/0.4006 | 0.6004/0.4117 | 0.6050/0.4134 | 0.5145/0.2654 | 0.5929/0.4030 | 0.6172/0.4116 | 0.5556/0.2971 |
| sit | 0.7201/0.7215 | 0.7112/0.7116 | 0.7443/0.7294 | 0.3601/0.3914 | 0.7341/0.7341 | 0.7354/0.7340 | 0.4720/0.5065 |
| stand up | 0.6174/0.6800 | 0.6362/0.6709 | 0.6240/0.6726 | 0.3561/0.4261 | 0.6428/0.6953 | 0.6384/0.6997 | 0.4267/0.4946 |
| bend | 0.2826/0.3199 | 0.2955/0.3147 | 0.3063/0.3227 | 0.3053/0.1964 | 0.2806/0.3159 | 0.2796/0.3189 | 0.2994/0.2056 |
| crawl | 0.7354/0.7673 | 0.6973/0.7318 | 0.7197/0.7571 | 0.3274/0.4384 | 0.7399/0.7639 | 0.7354/0.7610 | 0.4238/0.5455 |
| stumble | 0.1602/0.2373 | 0.1413/0.2160 | 0.1437/0.2204 | 0.0247/0.0472 | 0.1567/0.2364 | 0.1614/0.2414 | 0.0624/0.1122 |
| clap | 0.3930/0.4202 | 0.3765/0.4031 | 0.3868/0.4114 | 0.1193/0.1671 | 0.3786/0.3966 | 0.3498/0.3825 | 0.1502/0.2017 |
| throw | 0.5717/0.5009 | 0.5418/0.4836 | 0.5552/0.5012 | 0.2477/0.3156 | 0.5604/0.4877 | 0.5655/0.4986 | 0.3313/0.3874 |
| kick | 0.7543/0.5768 | 0.7092/0.5611 | 0.7135/0.5663 | 0.4497/0.3864 | 0.7431/0.5863 | 0.7578/0.5845 | 0.5052/0.4467 |
| knock | 0.1625/0.2462 | 0.1422/0.2104 | 0.1422/0.2150 | 0.0632/0.1061 | 0.1377/0.2075 | 0.1625/0.2445 | 0.0835/0.1348 |
| punch | 0.1749/0.2102 | 0.1665/0.2061 | 0.1665/0.2031 | 0.0754/0.1073 | 0.1738/0.2046 | 0.1780/0.2151 | 0.0890/0.1241 |
| touching face | 0.2141/0.2564 | 0.2301/0.2638 | 0.2510/0.2887 | 0.1504/0.1915 | 0.2112/0.2513 | 0.2241/0.2644 | 0.1763/0.2166 |

Best Stay-aware branch: `margin_listwise` (0.458730 Acc, 0.446342 Macro-F1; ΔAcc +0.004464).
AnyCorrect Oracle ceiling: 0.728274; remaining gap from Frozen: 0.274008.
The baseline reproduction passed the ±0.5pp gate.

Scientific decision: if no Stay-aware branch exceeds FrozenStageCv0 by 2pp, the previous failure is not explained by stay/candidate calibration alone; the remaining bottleneck is observable H1 representation. No new method is started automatically.

No Test files were read and no skeleton/RGB/DINO/perception data were regenerated.
