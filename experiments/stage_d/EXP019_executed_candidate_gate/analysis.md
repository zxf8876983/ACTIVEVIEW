# EXP019 analysis

## Protocol and provenance

EXP019 trained exactly one `Linear(12,64) → ReLU → Linear(64,1)` binary gate
for 30 fixed Train epochs with `BCEWithLogitsLoss`, Adam, learning rate
`1e-3`, batch size `256`, and seed `42`. The input was the normalized legal
Stage-D geometry for the frozen EXP014-selected candidate `c_hat`, appended
with its frozen predicted utility. The target was `1[true_U2(c_hat) > 0]` and
was used only for Train supervision/offline Val diagnostics. Stage C-v0's
first action/p1 and EXP014's candidate ranking were unchanged; Val used the
fixed `sigmoid(logit) > 0.5` rule once. No Test data were read.

Full runtime evidence is under:

`ACTIVEVIEW_DATA_ROOT/experiments/stage_d/EXP019_executed_candidate_gate/`

The compact source result is `result.json`; the result records hashes for the
Stage-D cache, Train/Val frozen predictions, Stage-B Val utility and final
gate checkpoint.

## Train calibration/training

| quantity | value |
|---|---:|
| Train eligible episodes | 29,133 |
| positive `y_exec` | 12,877 |
| negative `y_exec` | 16,256 |
| final BCE loss (epoch 30) | 0.652773 |

There was no Val-based early stopping or selection; the final epoch was used.

## Val gate diagnostics (`y_exec`)

The Val gate set contains 9,742 frozen-v0-Move episodes. Positive prevalence
is 0.439437 and the gate predicted Move for 0.316875 of episodes.

| metric | value |
|---|---:|
| gate accuracy | 0.581605 |
| balanced accuracy | 0.560309 |
| Move precision / recall / F1 | 0.533204 / 0.384490 / 0.446797 |
| Stay precision / recall | 0.604057 / 0.736129 |
| ROC-AUC / PR-AUC | 0.615475 / 0.535490 |

Confusion counts are: learned Stay/oracle Stay `4020`, learned Stay/oracle
Move `2635`, learned Move/oracle Stay `1441`, and learned Move/oracle Move
`1646`.

## Canonical Val trajectory metrics

All rows use the existing Stage-D trajectory evaluator and the same frozen
first-step protocol.

| Variant | Gate | Candidate | Accuracy | Macro-F1 | Mean regret | Median regret | P90 regret | Headroom | Avg moves | Mean geodesic (m) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP014 | predicted utility > 0 | frozen learned `c_hat` | 0.658254 | 0.610153 | 1.422463 | 0.003526 | 5.515663 | 0.783313 | 0.864946 | 2.522080 |
| EXP019 | learned gate p > 0.5 | frozen learned `c_hat` | 0.656681 | 0.607936 | 1.429851 | 0.003841 | 5.532555 | 0.780325 | 0.917209 | 2.554839 |
| ExecutedCandidateOracleGate | true `U2(c_hat)` > 0 | frozen learned `c_hat` | 0.743119 | 0.693231 | 0.761339 | 0.000024 | 2.831560 | 0.865193 | 1.002574 | 2.692688 |

The frozen EXP014 and executed-candidate oracle references match their
accepted values within `1e-5`. Candidate identity mismatches between EXP014
and EXP019 are `0`; EXP019 changes only the gate.

## Headroom recovery

Relative to EXP014, the executed-candidate oracle accuracy gap is `0.084865`
and the mean-regret gap is `0.661123`. EXP019 has an accuracy change of
`-0.001573` and a mean-regret change of `+0.007388` (worse). Thus its
accuracy headroom recovery is `-0.018534` and regret-gap recovery is
`-0.011176`.

## Action changes relative to EXP014

The gate produced `984` Stay→Move changes and `253` Move→Stay changes;
`6,402` episodes stayed in both policies and `2,103` moved in both. Among
Stay→Move changes, `490` had positive executed true utility and `494` were
non-positive; their mean executed utility was `-0.224513`. Among Move→Stay
changes, `122` were positive and `131` non-positive, with mean `-0.464739`.
The near balance of beneficial and harmful changes does not indicate a useful
correction of the frozen gate.

## Controlled interpretation

**Observation.** The small gate learned a modestly predictive signal
(ROC-AUC `0.615475`), but its fixed 0.5 Val decisions reduced trajectory
Accuracy from `0.658254` to `0.656681`, increased mean regret from `1.422463`
to `1.429851`, and reduced headroom capture from `0.783313` to `0.780325`.

**Interpretation.** Under this representation and fixed training protocol,
direct executed-candidate supervision did not transfer into better trajectory
decisions. The large executed-candidate oracle gap (`0.743119` Accuracy)
therefore remains an offline ceiling, not evidence that this minimal gate is
deployable. The result does not by itself falsify the target; it indicates
that the current gate representation/capacity or joint action formulation is
insufficient at the fixed decision rule.

**Decision: INCONCLUSIVE.** EXP019 does not support accepting the trained gate
as a policy. No Test evaluation was performed and no upstream frozen artifact
was changed.

**Next (human review only):** consider either a richer legal current/candidate
representation or a jointly trained Stay/p2/p3 action formulation. Do not
start EXP020 automatically.
