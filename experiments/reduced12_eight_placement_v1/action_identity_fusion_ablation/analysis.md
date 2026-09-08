# Reduced12 action identity fusion ablation

Train/Val only on the existing moving contexts. No policy Test was read; no WM/JR/skeleton/RGB/DINO artifact was modified or regenerated.

## Val Moving comparison

| Method | Accuracy | Macro-F1 | ΔAccuracy vs S1 | ΔMacro-F1 vs S1 |
|---|---:|---:|---:|---:|
| S1-only | 0.454266 | 0.444782 | 0.000 | 0.000 |
| Mean posterior | 0.450099 | 0.435032 | -0.417 | -0.975 |
| Product-of-Evidence | 0.431944 | 0.405530 | -2.232 | -3.925 |
| Confidence-select | 0.447718 | 0.433241 | -0.655 | -1.154 |
| ST-GCN feature-only MLP | 0.531052 | 0.555232 | 7.679 | 11.045 |
| ST-GCN feature + logp MLP | 0.529563 | 0.547400 | 7.530 | 10.262 |
| DINO-only MLP | 0.388988 | 0.417400 | -6.528 | -2.738 |
| Current all-feature MLP | 0.547619 | 0.566860 | 9.335 | 12.208 |

## Focus actions

| Method | bend Recall/F1 | stumble Recall/F1 | knock Recall/F1 | touching face Recall/F1 |
|---|---:|---:|---:|---:|
| S1-only | 0.283/0.320 | 0.160/0.237 | 0.163/0.246 | 0.214/0.256 |
| Mean posterior | 0.287/0.321 | 0.104/0.174 | 0.165/0.254 | 0.214/0.258 |
| Product-of-Evidence | 0.358/0.317 | 0.052/0.096 | 0.077/0.135 | 0.224/0.254 |
| Confidence-select | 0.271/0.312 | 0.113/0.186 | 0.167/0.257 | 0.220/0.263 |
| ST-GCN feature-only MLP | 0.307/0.333 | 0.344/0.368 | 0.713/0.709 | 0.278/0.266 |
| ST-GCN feature + logp MLP | 0.307/0.339 | 0.398/0.387 | 0.657/0.635 | 0.220/0.270 |
| DINO-only MLP | 0.259/0.266 | 0.264/0.261 | 0.711/0.717 | 0.267/0.283 |
| Current all-feature MLP | 0.319/0.365 | 0.362/0.379 | 0.770/0.745 | 0.289/0.321 |

## Scientific interpretation

No analytic posterior fusion branch improves S1-only Macro-F1.
ST-GCN feature-only reaches 0.531052/0.555232; adding logp changes it to 0.529563/0.547400.
DINO is complementary in the all-feature branch (all-feature Macro-F1 0.566860 vs feature+logp 0.547400); DINO-only is 0.417400.
The best Val-moving branch is Current all-feature MLP with Accuracy/Macro-F1 0.547619/0.566860; this is a single-seed Val comparison, not Test evidence.
Current all-feature MLP reaches 0.547619/0.566860; compare this with the prior approximately 55% identity result without claiming generalization beyond Val.

## Protocol

- taxonomy: reduced12 (walk, sit, stand up, bend, crawl, stumble, clap, throw, kick, knock, punch, touching face)
- MLP: two layers, hidden=256, GELU, CrossEntropy, seed=42, 20 epochs, best moving-Val Macro-F1
- normalization: Train-only mean/std for each MLP branch
- `test_used=false`; no Test path or artifact was read
