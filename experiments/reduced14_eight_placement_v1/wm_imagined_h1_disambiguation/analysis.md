# Reduced14 WM-imagined H1 disambiguation (Val)

Val moving contexts: 14809; legal H1 candidate hypotheses: 441283. Test was not read.

## Final history identity after real-observation evaluation

| Selector | Accuracy | Macro-F1 | Mean entropy | s0-error correction |
|---|---:|---:|---:|---:|
| Frozen_current_H1 | 0.482815 | 0.496501 | 1.445789 | 0.405738 |
| Privileged_real_Min_entropy_H1 | 0.512594 | 0.504617 | 0.734740 | 0.432696 |
| Privileged_real_Max_margin_H1 | 0.515970 | 0.510083 | 0.772546 | 0.436157 |
| Old_WM_imagined_Min_entropy_H1 | 0.345668 | 0.365998 | 1.656082 | 0.249636 |
| Old_WM_imagined_Max_margin_H1 | 0.327166 | 0.346189 | 1.702879 | 0.235246 |
| Ranking_aware_WM_imagined_Min_entropy_H1 | 0.395840 | 0.404250 | 1.517553 | 0.324226 |
| Ranking_aware_WM_imagined_Max_margin_H1 | 0.394220 | 0.402601 | 1.525719 | 0.321129 |
| IdentityOracle_H1 | 0.899588 | 0.900413 | 1.047738 | 0.870036 |

## Imagined-versus-real belief alignment

- **Old_WM** entropy Pearson/Spearman = 0.559986/0.477783; margin Pearson/Spearman = 0.425398/0.265875; min-entropy/max-margin candidate overlap = 0.048281/0.042001.
- **Ranking_aware_WM** entropy Pearson/Spearman = 0.531035/0.528160; margin Pearson/Spearman = 0.361194/0.295388; min-entropy/max-margin candidate overlap = 0.052941/0.050577.

## Interpretation

Old WM-E imagined min-entropy changes Frozen H1 Accuracy by -0.137146; ranking-aware WM-E changes it by -0.086974.
This is an H=1 s0-only deployment diagnostic against WM-E checkpoints trained with H=2 histories; the fixed interface shift is documented above. Imagined selectors are evaluated with the selected candidate's real archived observation, so their final identity metrics do not leak imagined labels or use candidate observations as WM inputs.
If imagined selectors remain below Frozen H1 and privileged real-observation selectors, the limiting factor is WM-E action-discriminative representation fidelity; the next step should improve that objective rather than train another scalar H1 ranker.
Leakage audit: `test_used=false`; no model was trained, no formal checkpoint was modified, and only Val archives/caches were read.
