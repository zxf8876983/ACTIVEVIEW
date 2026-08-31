# EXP025 — DINOv2 Spatial RGB Utility Regression

Val-only result (13,987 episodes; 9,742 frozen-v0-Move second-step episodes).
The frozen EXP014 candidate ranking and first-step protocol were unchanged;
candidate identity mismatches were zero and no future-candidate RGB was read.

| Variant | Accuracy | Macro-F1 | Mean regret | P90 regret | Headroom |
|---|---:|---:|---:|---:|---:|
| EXP014 | 0.658254 | 0.610153 | 1.422463 | 5.515663 | 0.783313 |
| EXP022 | 0.659898 | 0.611687 | 1.416495 | 5.494913 | 0.782352 |
| EXP024 Global CLS | 0.657682 | 0.609503 | 1.414353 | 5.431195 | 0.782434 |
| EXP025 Spatial RGB | 0.659898 | 0.607331 | 1.378650 | 5.330633 | 0.784692 |
| ExecutedCandidateOracle | 0.743119 | 0.693231 | 0.761339 | 2.831560 | 0.865193 |

EXP025 minus EXP024: Accuracy +0.002216, Macro-F1 -0.002172, mean regret
-0.0357029, and P90 regret -0.100562. Utility regression MAE/RMSE were
2.839231/4.311770, with Pearson 0.439911 and Spearman 0.266806. Spatial RGB
improves regret modestly over Global CLS, but trajectory metrics are not
uniformly improved; decision: **INCONCLUSIVE**.

The DINOv2 spatial cache contains 58,266 Train and 19,484 Val observations
(77,750 unique \`s0\`/\`s1\` keys), float16 \`[16,768]\` tokens, and no p2/p3 RGB.
