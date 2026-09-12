# Historical Route-1 × Shared Adapted Head Synergy Audit

## Historical method

No deployment-legal reduced12 matched-protocol historical Route-1 method above 50% was found. The closest compliant method is **Stay-aware GTMargin Listwise (OldBestOneShot)**, with the archived historical result 0.458730 Acc / 0.446342 Macro-F1. Its checkpoint was replayed on the current 10,080-context matched population.

The 0.502778 RealEvidence-GTMarginListwise result is excluded because it consumes future candidate evidence; old 16-class EXP036 results are protocol mismatches.

## Same selected viewpoints

| Policy / recognizer | Accuracy | Macro-F1 | Move rate |
|---|---:|---:|---:|
| Stay/current + Original | 0.254266 | 0.235500 | 0.000000 |
| Stay/current + Shared | 0.302579 | 0.292976 | 0.000000 |
| Historical strong Route-1 + Original | 0.458730 | 0.446342 | 0.972024 |
| Historical strong Route-1 + Shared | 0.509524 | 0.509806 | 0.972024 |

## Additive decomposition

- Δ recognizer at Stay: **+0.048313**
- Δ policy under Original: **+0.204464**
- Δ policy under Shared: **+0.206944**
- Δ combined (Historical+Shared − Stay+Original): **+0.255258**
- Expected additive accuracy: **0.507044**
- Synergy: **+0.002480**

## Context transitions

For both Stay and the historical selected viewpoint, `result.json` gives old-wrong→shared-correct correction, old-correct→shared-wrong regression, both-correct and both-wrong counts.

## Shared recognizer privileged references

GT-TrueLogP Oracle: 0.753175 Acc / 0.755358 Macro-F1; Legal AnyCorrect Coverage (stay + candidates): 0.791964; candidate-only coverage: 0.771726.

## Decision

Historical policy + Shared head clears the +2pp gate. This audit does not retrain the policy automatically; a separately approved second-stage replay would be required.

No Policy Test files were read. No recognizer, policy architecture, data, RGB, skeleton or DINO artifact was modified.
