# EXP026 — Spatial RGB-D Utility Regression

Val-only result (13,987 episodes; 9,742 frozen-v0-Move second-step episodes).
The frozen EXP014 candidate ranking and first-step protocol were unchanged;
candidate identity mismatches were zero. Depth was rendered only for visited
\`s0\`/\`s1\` observations at frame 15 using Habitat metric depth.

| Variant | Accuracy | Macro-F1 | Mean regret | P90 regret | Headroom |
|---|---:|---:|---:|---:|---:|
| EXP014 | 0.658254 | 0.610153 | 1.422463 | 5.515663 | 0.783313 |
| EXP022 | 0.659898 | 0.611687 | 1.416495 | 5.494913 | 0.782352 |
| EXP024 Global CLS | 0.657682 | 0.609503 | 1.414353 | 5.431195 | 0.782434 |
| EXP025 Spatial RGB | 0.659898 | 0.607331 | 1.378650 | 5.330633 | 0.784692 |
| EXP026 Spatial RGB-D | 0.657325 | 0.606847 | 1.407385 | 5.423377 | 0.783227 |
| ExecutedCandidateOracle | 0.743119 | 0.693231 | 0.761339 | 2.831560 | 0.865193 |

EXP026 minus EXP025: Accuracy -0.002574, Macro-F1 -0.000484, mean regret
+0.0287345, and P90 regret +0.0927432. Utility regression MAE/RMSE were
2.838240/4.305187, with Pearson 0.428920 and Spearman 0.248698. Depth did not
improve the spatial-RGB pilot; decision: **INCONCLUSIVE**.

The compact depth cache contains 58,266 Train and 19,484 Val observations
(77,750 unique \`s0\`/\`s1\` keys), \`[16,4]\` float16 features, 22,669,849 bytes
across cache files, and was generated with 16 workers and four-camera batching.
\`max_clipping_rate_gt_10m\` is unavailable because raw depth was intentionally
not retained. \`future_candidate_depth_used=false\`.
