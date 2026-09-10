# Reduced12 utility landscape locality

Moving Val contexts: 10080

## Adjacency

| Pair set | Pairs | Pearson | Spearman | P(neighbor correct | center correct) | P(neighbor correct | center wrong) | Mean |margin diff| |
|---|---:|---:|---:|---:|---:|---:|
| angular | 18158 | 0.468436 | 0.466105 | 0.588235 | 0.204551 | 5.436727 |
| radial | 27920 | 0.685202 | 0.688240 | 0.660790 | 0.133588 | 4.201709 |
| all_1hop | 46078 | 0.600455 | 0.600908 | 0.634455 | 0.162854 | 4.688393 |

## Local oracle ceiling

| Selector | Accuracy | Macro-F1 |
|---|---:|---:|
| H1 | 0.454266 | 0.444782 |
| 1-Hop Oracle | 0.538790 | 0.529863 |
| 2-Hop Oracle | 0.597222 | 0.591727 |
| Full GT-margin Oracle | 0.728274 | 0.722059 |
| AnyCorrect Oracle | 0.728274 | 0.729637 |

GT-margin-best distance: {'h1_itself': 0.2896825396825397, 'one_hop': 0.17688492063492064, 'within_two_hops': 0.13839285714285715, 'over_two_hops': 0.3950396825396825}
2-hop oracle to full GT-margin oracle accuracy gap: 13.105 pp.

This is a Val-only oracle audit; no policy Test, training, or perception regeneration was used.
