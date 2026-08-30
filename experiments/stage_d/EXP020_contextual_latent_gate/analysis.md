# EXP020 analysis

EXP020 used the frozen EXP014 ranker as an inference-only feature extractor.
The 128-D candidate token was taken immediately before `utility_head`, then
concatenated with the frozen predicted utility (129-D). Only the fixed
`Linear(129,64) → GELU → Linear(64,1)` gate was trained for 30 Train epochs
(Adam, lr `1e-3`, batch `256`, seed `42`); Val used the fixed probability
threshold `0.5` once. No Test, perception, Habitat or ST-GCN processing was
performed.

Runtime evidence is under:

`ACTIVEVIEW_DATA_ROOT/experiments/stage_d/EXP020_contextual_latent_gate/`

## Train

| quantity | value |
|---|---:|
| eligible episodes | 29,133 |
| positive / negative `y_exec` | 12,877 / 16,256 |
| final BCE loss | 0.626943 |

## Val gate diagnostics (`y_exec`)

The 9,742 Val v0-Move episodes had positive prevalence `0.439437`.

| metric | value |
|---|---:|
| Accuracy | 0.599056 |
| Balanced Accuracy | 0.588013 |
| Move precision / recall / F1 | 0.548337 / 0.496847 / 0.521324 |
| Stay precision / recall | 0.632611 / 0.679180 |
| ROC-AUC / PR-AUC | 0.641344 / 0.554610 |
| Move rate | 0.398173 |

Confusion counts: learned Stay/oracle Stay `3709`, learned Stay/oracle Move
`2154`, learned Move/oracle Stay `1752`, learned Move/oracle Move `2127`.

## Val trajectory metrics

| Variant | Accuracy | Macro-F1 | Mean regret | Median | P90 | Headroom | Avg moves | Mean geodesic (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP014 | 0.658254 | 0.610153 | 1.422463 | 0.003526 | 5.515663 | 0.783313 | 0.864946 | 2.522080 |
| EXP019 | 0.656681 | 0.607936 | 1.429851 | 0.003841 | 5.532555 | 0.780325 | 0.917209 | 2.554839 |
| EXP020 | 0.661757 | 0.612956 | 1.453868 | 0.002727 | 5.635810 | 0.778625 | 0.973833 | 2.640947 |
| ExecutedCandidateOracle | 0.743119 | 0.693231 | 0.761339 | 0.000024 | 2.831560 | 0.865193 | 1.002574 | 2.692688 |

EXP020 gains `+0.003503` Accuracy over EXP014, recovering `4.13%` of the
executed-candidate oracle Accuracy gap. However mean regret increases by
`0.031405`, corresponding to `-4.75%` regret-gap recovery; P90 regret also
worsens. Candidate identity mismatch count is zero, so the intervention only
changes the gate.

Relative to EXP014, EXP020 changes `1855` Stay→Move and `332` Move→Stay
decisions. Stay→Move contains `994` positive and `861` non-positive executed
utilities (mean `-0.224564`); Move→Stay contains `145` positive and `187`
non-positive cases (mean `0.068378`).

## Representation comparison

| Model | Representation | ROC-AUC | PR-AUC | Trajectory Accuracy |
|---|---|---:|---:|---:|
| EXP019 | normalized 11-D geometry + predicted utility | 0.615475 | 0.535490 | 0.656681 |
| EXP020 | frozen EXP014 contextual token + predicted utility | 0.641344 | 0.554610 | 0.661757 |

**Observation.** The contextual latent improves gate discrimination and
Accuracy modestly over EXP019, but increases trajectory mean/P90 regret and
reduces headroom capture relative to EXP014.

**Interpretation.** The frozen EXP014 contextual representation contains more
information for `y_exec` than the local 12-D representation, but the fixed
binary gate remains poorly aligned with trajectory utility. This is evidence
for representation-level signal, not evidence that the gate is deployable.

**Decision: INCONCLUSIVE.** No policy is accepted; Test remains locked.

**Next (human review only):** assess a jointly trained Stay/p2/p3 objective or
another legal gate representation. Do not start EXP022 automatically.
