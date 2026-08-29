# EXP001 Analysis — Utility-Gap-Aware Ranking

## Observation

EXP001 used the frozen Stage C-v0 feature, architecture and protocol, adding
only the utility-gap-aware pairwise ranking term (`lambda_gap=1.0`,
`tau_gap=1.0`, `max_gap_weight=10.0`). It trained on Train and selected the
checkpoint by Val Macro-F1 (epoch 28 of 38; patience 10). Test was not used.

On the 13,987 Val Episodes:

| Metric | Stage C-v0 | EXP001 | Change |
|---|---:|---:|---:|
| Accuracy | 64.9103% | 64.6100% | -0.30 pp |
| Macro-F1 | 59.8042% | 59.9520% | +0.15 pp |
| Mean regret | 1.4505 | 1.4646 | +0.0141 |
| Median regret | 0.00514 | 0.00613 | +0.00099 |
| P90 regret | 5.6078 | 5.6962 | +0.0884 |
| Positive headroom capture | 77.7965% | 77.5327% | -0.26 pp |
| C2 rate | 31.7438% | 32.1012% | +0.36 pp |
| Large-gap mean regret | 2.4493 | 2.4414 | 0.32% relative improvement |

Failure taxonomy for EXP001 was C2 wrong high-loss candidate 32.10%, missed
move 20.02%, C1 near-optimal wrong candidate 11.30%, unnecessary move 5.84%,
and correct SafeOracle action 30.74%. Regret groups were near-optimal 43.67%,
low 31.33%, moderate 15.00% and high 10.00%; the top-5% tail had mean regret
10.7745. The top-5% proportion is definition-induced and is not a future
performance metric.

Diagnostic evidence remains consistent with the frozen failure analysis:
`play instrument`, `stumble` and `knock` are the most difficult classes;
high-regret rates across regions were bedroom 9.10%, kitchen 9.54%,
living_room 9.92% and dining_area 11.25%, with no strong region-specific
pattern. Regret correlations were weak for entropy (Spearman 0.175) and pose
confidence (0.059). The learned policy selected closer candidates more often
(approximately 79% versus 55% for SafeOracle). Symmetric-geometry ambiguity
covered 2.83% of Episodes and had only 1.11× high-regret enrichment.

The worst 10% of motion records contributed 37.43% of catastrophic failures,
approximately 3.74× the uniform share.

## Interpretation

Under the tested formulation and weight setting, explicitly increasing the
penalty for large ground-truth utility-gap ordering errors did not materially
improve harmful viewpoint-selection errors. The Stage C-v0 failure mode
therefore does not appear to be explained primarily by insufficient pairwise
utility-gap weighting in the training objective. This result does not establish
that every gap-aware loss is ineffective; it is specific to this preregistered
formulation and setting.

The stronger current evidence is concentration in difficult motion records,
not a single entropy/confidence signal or a robust body-orientation ambiguity.

## Decision

**REJECT** — the 5% large-gap improvement criterion and the required secondary
improvement were not met, although the Macro-F1 safety condition was met.

## Next Step

Prepare, but do not start, **EXP002 — Train-only hard-record-aware sampling**.
