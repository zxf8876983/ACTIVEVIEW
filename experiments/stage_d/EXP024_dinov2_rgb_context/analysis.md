# EXP024 analysis

EXP024 uses only already visited Stage-D s0/s1 RGB observations. DINOv2
ViT-B/14 is frozen; only the shared RGB projector and 513-D utility
regression head are trained on Train. Val is evaluated once and Test is locked.

## RGB audit

- Unique Train RGB observations: 58266
- Unique Val RGB observations: 19484
- Cache hits / misses: 0 / 77750
- Cache extraction time (s): 1354.320
- Cache disk bytes: 141402204
- Future-candidate RGB used: false

## Train

- Episodes: 29133
- Final SmoothL1 loss: 1.456738

## Val regression/sign diagnostics

MAE / RMSE: 2.896349 / 4.314998
Pearson / Spearman: 0.428491057722588 / 0.25233485753849866
Sign accuracy / balanced accuracy: 0.574420 / 0.566241
ROC-AUC / PR-AUC: 0.6067992010279855 / 0.5391943684142505

## Val trajectory metrics

| Variant | Accuracy | Macro-F1 | Mean regret | Median | P90 | Headroom | Avg moves | Mean geodesic (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP014 | 0.658254 | 0.610153 | 1.422463 | 0.003526 | 5.515663 | 0.783313 | 0.864946 | 2.522080 |
| EXP022 | 0.659898 | 0.611687 | 1.416495 | 0.003513 | 5.494913 | 0.782352 | 0.908057 | 2.560522 |
| EXP024 | 0.657682 | 0.609503 | 1.414353 | 0.004255 | 5.431195 | 0.782434 | 0.992136 | 2.689528 |
| ExecutedCandidateOracle | 0.743119 | 0.693231 | 0.761339 | 0.000024 | 2.831560 | 0.865193 | 1.002574 | 2.692688 |

## EXP024 vs EXP022

- Accuracy delta: -0.002216
- Mean-regret delta (EXP024 - EXP022): -0.002142
- Accuracy recovery versus ExecutedCandidateOracle: -0.006739679865207438
- Mean-regret recovery versus ExecutedCandidateOracle: 0.012266494553169022

## Scientific interpretation

This is a diagnostic RGB-global-embedding result, not a deployment
acceptance decision. Any gain or loss is interpreted relative to EXP022
without changing the frozen candidate ranking or first-step protocol.
