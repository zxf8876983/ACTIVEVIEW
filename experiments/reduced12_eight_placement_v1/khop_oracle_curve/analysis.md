# Reduced12 K-hop reachable oracle curve

Moving Val contexts: 10080

## K-hop reachability oracle

| Hop | Accuracy | Macro-F1 | AnyCorrect coverage | ΔAcc vs previous | Full gap |
|---|---:|---:|---:|---:|---:|
| H1 | 0.454266 | 0.444782 | 0.454266 | 0.000000 | 0.255357 |
| K1 | 0.538790 | 0.529863 | 0.538790 | 0.084524 | 0.170833 |
| K2 | 0.597222 | 0.591727 | 0.597222 | 0.058433 | 0.112401 |
| K3 | 0.654762 | 0.650562 | 0.654762 | 0.057540 | 0.054861 |
| K4 | 0.688393 | 0.683594 | 0.688393 | 0.033631 | 0.021230 |
| K5 | 0.701984 | 0.697100 | 0.701984 | 0.013591 | 0.007639 |
| K6 | 0.708234 | 0.702974 | 0.708234 | 0.006250 | 0.001389 |
| Full | 0.709623 | 0.704598 | 0.709623 | 0.000000 | 0.000000 |

## Greedy local oracle

| Selector | Accuracy | Macro-F1 | Move rate |
|---|---:|---:|---:|
| GreedyOracle-Step1 | 0.538790 | 0.529863 | 0.391964 |
| GreedyOracle-Step2 | 0.554663 | 0.547077 | 0.091171 |
| GreedyOracle-Step3 | 0.557937 | 0.550176 | 0.018254 |
| GreedyOracle-Step4 | 0.558234 | 0.550490 | 0.002579 |

Minimum hop to any correct candidate: {'0': 0.454265873015873, '1': 0.08452380952380953, '2': 0.05843253968253968, '3': 0.057539682539682536, '4': 0.03363095238095238, '5+': 0.02123015873015873, 'unreachable/none': 0.29037698412698415}
Minimum hop to GT-margin-best candidate: {'0': 0.2896825396825397, '1': 0.17688492063492064, '2': 0.13839285714285715, '3': 0.1665674603174603, '4': 0.12341269841269842, '5+': 0.10505952380952381, 'unreachable/none': 0.0}

## Correct-view basins

Contexts with a correct candidate: 0.709623; mean largest correct component: 2.2289; mean fraction in largest component: 0.763210.

The path oracle is the K-hop reachability ceiling; greedy results show whether monotonic local exploration can attain that ceiling. This is a Val-only privileged diagnostic and did not read policy Test or generate data.
