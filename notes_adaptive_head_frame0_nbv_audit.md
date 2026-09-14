# Adaptive Head × Frame0 NBV Combination Audit (2026-09-14)

## Scope

- Train contexts: 46,324; Moving Val contexts: 10,080; records: 313/105.
- Action set: current/Stay plus Stage-A legal candidate pool.
- Strict protocol: current frame-0 RGB/DINO and legal geometry select one view;
  selected real O1 is classified alone.
- No Policy Test, RGB/skeleton/DINO regeneration, encoder modification or
  multi-view fusion.

## Heads

| Head | Val s1 Accuracy/F1 | Legal-candidate Accuracy/F1 |
|---|---:|---:|
| H_shared | 0.507044 / 0.507894 | 0.375418 / 0.378473 |
| H_adapt_old | 0.530952 / 0.555852 | 0.369800 / 0.392751 |
| H_adapt_balanced | 0.488393 / 0.493137 | 0.364633 / 0.367584 |

The old adaptive head is strongly s1-specialized. Candidate-only AnyCorrect
coverage is 0.771726, 0.785417 and 0.747123 for shared, old and balanced.

## Strict Frame0 selectors

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Random + H_shared | 0.365079 | 0.364567 |
| Historical selector + H_shared | 0.506151 | 0.503722 |
| Historical selector + H_adapt_old | 0.523115 | 0.542017 |
| Retrained old-adaptive selector | 0.526984 | 0.547002 |
| Retrained balanced-adaptive selector | 0.493452 | 0.497652 |

Gain decomposition: head gain -1.091pp, historical shared-selector gain
+14.107pp, adaptive-aware selector gain +13.929pp, combined gain +12.837pp,
synergy -0.179pp. Shared/old and shared/balanced candidate utility rankings
have Spearman 0.908 and 0.954.

## Decision

The balanced legal-candidate head does not generalize and the adaptive-aware
strict selector is below the historical selector by 1.270pp Accuracy. Under
the preregistered threshold this is **KILL ADAPTIVE-HEAD × NBV COMBINATION**.
The old head is retained only as a matched s1 diagnostic.

Report: `experiments/reduced12_eight_placement_v1/adaptive_head_frame0_nbv_audit/`.
