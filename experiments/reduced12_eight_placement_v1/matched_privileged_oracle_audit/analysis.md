# Matched Privileged Oracle Audit

Moving Val only (10,080 contexts). The formal action set is each context's separate stay/current action plus its Stage-A legal reachable candidate pool; all-32 is reported only as an adapted-head coverage diagnostic.

## Action-set alignment

Candidate actions (excluding stay): mean/min/max=6.816/2/21; legal actions including stay: mean/min/max=7.816/3/22. Stage-A/Stage-C/cache alignment errors: 0.

## Overall metrics

| Recognizer / method | Accuracy | Macro-F1 | Move rate |
|---|---:|---:|---:|
| Frozen original ST-GCN / Current/Stay | 0.254266 | 0.235500 | 0.000000 |
| Frozen original ST-GCN / GT-TrueLogP Oracle | 0.727976 | 0.721495 | 0.884425 |
| Frozen original ST-GCN / GTMargin Oracle | 0.728274 | 0.722059 | 0.887401 |
| Adapted single-view head / Current/Stay | 0.285714 | 0.293422 | 0.000000 |
| Adapted single-view head / GT-TrueLogP Oracle | 0.774802 | 0.788596 | 0.885020 |
| Adapted single-view head / GTMargin Oracle | 0.803968 | 0.820770 | 0.884425 |
| Adapted single-view head / MaxConfidence | 0.534623 | 0.539143 | 0.875198 |

Frozen legal AnyCorrect Coverage: 0.728274 (7341/10080). Adapted legal AnyCorrect Coverage: 0.803968 (8104/10080). Adapted unrestricted all-32 AnyCorrect Coverage: 0.957540 (9652/10080).

## Formal oracle interpretation

The frozen GT-TrueLogP legal oracle Accuracy is 0.727976; compared with the historical ~72.8% reference, the discrepancy is -0.030pp. Frozen GTMargin is 0.728274. Because the action set and final prediction rule are matched, this is the formal privileged H0 oracle for the frozen recognizer. AnyCorrect is coverage only, not Oracle Accuracy.

The adapted-head legal oracle uses the same action set and actual selected-view argmax; its MaxConfidence row is a no-GT selector. The unrestricted all-32 adapted number is not a formal policy oracle and should not be mixed with the legal-action result.

## Historical/protocol audit

The 95.754% adapted all-32 coverage (if reproduced) is inflated relative to the formal legal action space because it permits every lattice viewpoint rather than Stage-A reachable candidates. Legal candidate count and stay/current alignment are therefore reported explicitly.

## Leakage and boundaries

- `policy_test_used=false`; only Stage-A/Stage-C/Stage-D Val and Val counterfactual/archive artifacts were opened.
- No model was trained, no selector was trained, and no RGB/skeleton/perception data was generated.
- GT label and true candidate logp are used only for privileged oracle selection/coverage; final predictions always come from the selected recognizer output.
- Future archived skeletons are used for terminal evidence and adapted all-32 diagnostic inference only; they are not deployable policy inputs.
