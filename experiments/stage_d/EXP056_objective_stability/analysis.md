# EXP056 analysis

The paired three-seed run used the same 29,133 Train contexts, frozen WM-E and
candidate graph, `_JointRevision` architecture, optimizer (AdamW, 1e-3,
1e-4), batch size 512, 20 epochs, and real-observation H2 evaluator.  Only the
main objective differs: ORIGINAL_JR uses the first correct action CE target;
MULTI_POSITIVE_JR uses the set log-sum-exp objective with the same correctness
BCE (0.25) and posterior CE (0.05).

## Moving subset (9,742)

| Seed | Original Acc | Multi Acc | ΔAcc | Original F1 | Multi F1 | ΔF1 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.666085 | 0.683022 | +0.016937 | 0.637096 | 0.647338 | +0.010243 |
| 43 | 0.665059 | 0.683022 | +0.017963 | 0.634351 | 0.647443 | +0.013092 |
| 44 | 0.666598 | 0.691439 | +0.024841 | 0.640779 | 0.657359 | +0.016580 |

Means (population standard deviation): Original Acc `0.665914 ± 0.000640`,
Multi Acc `0.685828 ± 0.003968`, paired ΔAcc `+0.019914 ± 0.003509`;
Original F1 `0.637408 ± 0.002634`, Multi F1 `0.650713 ± 0.004699`, paired
ΔF1 `+0.013305 ± 0.002591`.

## Full population (13,987)

| Seed | Original Acc | Multi Acc | ΔAcc | Original F1 | Multi F1 | ΔF1 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.688925 | 0.700722 | +0.011797 | 0.646425 | 0.650955 | +0.004530 |
| 43 | 0.688210 | 0.700722 | +0.012512 | 0.643917 | 0.651062 | +0.007145 |
| 44 | 0.689283 | 0.706585 | +0.017302 | 0.649417 | 0.659162 | +0.009745 |

Means: Original Acc `0.688806 ± 0.000446`, Multi Acc `0.702676 ± 0.002764`,
paired ΔAcc `+0.013870 ± 0.002444`; Original F1 `0.646586 ± 0.002248`, Multi
F1 `0.653726 ± 0.003844`, paired ΔF1 `+0.007140 ± 0.002129`.

Multi-positive wins all 3/3 seeds for both Accuracy and Macro-F1.  Paired
terminal audits (rescued / harmful / net) were 925/760/+165 (seed 42),
948/773/+175 (seed 43), and 901/659/+242 (seed 44).  Stage-C first actions
changed in zero cases in every audit.

## Decision

**CASE A — STABLE PASS.** Multi-positive improves both Accuracy and Macro-F1
for all three seeds on the paired moving subset, with positive mean deltas and
the same direction on the full population.  The EXP055 multi-positive
objective is therefore recorded as the recommended canonical decision
objective, with fixed seed 42 as the recommended checkpoint.  This is Val-only
evidence; no Test split was accessed.

`test_used=false`; no perception regeneration, Habitat rendering, or ST-GCN
retraining was performed.
