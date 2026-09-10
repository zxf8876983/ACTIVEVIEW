# Reduced12 diverse Top-K proposal audit

Moving Val contexts: 10080

| Shortlist | AnyCorrect coverage | Oracle Acc | Oracle Macro-F1 | Mean azimuth separation | Mean lattice distance |
|---|---:|---:|---:|---:|---:|
| Ordinary Top1 | 0.454266 | 0.538294 | 0.523166 | 0.000 | 0.000 |
| Ordinary Top3 | 0.629861 | 0.664782 | 0.654544 | 66.132 | 2.278 |
| Ordinary Top5 | 0.678472 | 0.701984 | 0.694035 | 71.384 | 2.610 |
| Azimuth-Diverse Top1 | 0.454266 | 0.538294 | 0.523166 | 0.000 | 0.000 |
| Azimuth-Diverse Top3 | 0.629762 | 0.662202 | 0.652274 | 81.633 | 2.672 |
| Azimuth-Diverse Top5 | 0.678175 | 0.701687 | 0.692455 | 79.926 | 2.884 |
| Lattice-Diverse Top1 | 0.454266 | 0.538294 | 0.523166 | 0.000 | 0.000 |
| Lattice-Diverse Top3 | 0.626885 | 0.658631 | 0.648197 | 80.951 | 2.929 |
| Lattice-Diverse Top5 | 0.677877 | 0.700496 | 0.692117 | 77.934 | 3.004 |

Best diverse Top3 coverage delta vs ordinary: -0.010 pp.
No GT utility was used for shortlist ordering; true evidence is used only for oracle-within-shortlist evaluation.

policy_test_used=false; no training or perception regeneration.
