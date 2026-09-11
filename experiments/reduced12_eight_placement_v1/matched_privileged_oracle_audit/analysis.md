# Matched Privileged Viewpoint Oracle (fresh recomputation)

This report was recomputed by `run_reduced12_matched_privileged_oracle_audit.py` from runtime Val artifacts; no prior experiment `result.json` was read.

## Protocol and action set

Moving Val contexts: 10080. Each action set is exactly `stay/current + Stage-A candidate_pool`; candidate count mean/min/max=6.816/2/21, including-stay action count mean/min/max=7.816/3/22. Stage-A/Stage-C/cache action-set errors: 0.

## Required metrics

| Recognizer | Current/Stay Acc/F1 | GT-TrueLogP Acc/F1 | GT-Margin Acc/F1 | MaxConfidence Acc/F1 | Legal AnyCorrect Coverage |
|---|---:|---:|---:|---:|---:|
| Frozen | 0.254266/0.235500 | 0.727976/0.721495 | 0.728274/0.722059 | 0.461706/0.443670 | 0.728274 |
| Adapted | 0.285714/0.293422 | 0.774802/0.788596 | 0.803968/0.820770 | 0.534623/0.539143 | 0.803968 |

Adapted unrestricted-all32 AnyCorrect Coverage: 0.957540 (9652/10080); diagnostic-only, not the formal policy action set.

## Sanity and interpretation

Frozen GT-TrueLogP legal oracle differs from the historical 0.728274 reference by -0.030pp; GT-Margin is 0.728274. This is within the requested 1pp sanity bound, so action-set alignment is accepted. The formal privileged legal oracle should be reported as the actual selected-view GT-Margin/GT-TrueLogP result, while AnyCorrect is coverage only and is never presented as Oracle Accuracy.

GT-TrueLogP and GT-Margin always select the maximum score and then report that selected view's actual recognizer argmax; no correct-candidate shortcut is used. MaxConfidence is GT-free scoring but still reports the selected view's actual argmax.

The 0.957540 adapted all-32 coverage is protocol-inflated relative to the formal Stage-A reachable action space: allowing all 32 lattice viewpoints raises coverage from the legal 0.803968 to 0.957540.

## Boundaries

- `policy_test_used=false`; only Moving Val Stage-A/Stage-C/Stage-D and Val counterfactual/archive artifacts were read.
- `training_used=false`; no selector, recognizer, or checkpoint was trained or modified.
- No RGB, skeleton, DINO, or perception data was generated.
- Future archived skeletons were used only for frozen terminal evidence and the adapted all-32 diagnostic, never as deployable policy input.
