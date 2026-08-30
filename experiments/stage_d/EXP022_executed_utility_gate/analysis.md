# EXP022 analysis

EXP022 used the frozen EXP014 contextual candidate token immediately before
`utility_head`, concatenated with the frozen EXP014 predicted utility. Only a
fixed `Linear(129,64) → GELU → Linear(64,1)` regressor was trained for 30
Train epochs with default `SmoothL1Loss` (Adam, lr `1e-3`, batch `256`, seed
`42`). Val used the fixed strict sign rule `predicted_U_exec > 0` exactly
once. True U2 was a raw Train target/offline diagnostic only and never an
input. No Test, perception, Habitat or ST-GCN processing was performed.

Runtime evidence is under:

`ACTIVEVIEW_DATA_ROOT/experiments/stage_d/EXP022_executed_utility_gate/`

## Train

| quantity | value |
|---|---:|
| eligible episodes | 29,133 |
| positive / non-positive raw U2 targets | 12,877 / 16,256 |
| final Huber loss | 2.392657 |

## Val regression and sign diagnostics (9,742 v0-Move episodes)

| metric | value |
|---|---:|
| MAE / RMSE | 2.719621 / 4.305865 |
| Pearson / Spearman | 0.363494 / 0.248498 |
| sign Accuracy / Balanced Accuracy | 0.586122 / 0.563278 |
| Move precision / recall / F1 | 0.542075 / 0.374679 / 0.443094 |
| Stay precision / recall | 0.605337 / 0.751877 |
| ROC-AUC / PR-AUC | 0.612799 / 0.538928 |
| predicted Move rate | 0.303736 |

False Move: 1,355 episodes, mean true U2 `-2.728058`, median `-1.508636`,
total negative-utility magnitude `3,696.518`. False Stay: 2,677 episodes,
mean true U2 `2.042267`, median `0.579444`, total missed positive utility
`5,467.148`.

## Val trajectory metrics

| Variant | Accuracy | Macro-F1 | Mean regret | Median | P90 | Headroom | Avg moves | Mean geodesic (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP014 | 0.658254 | 0.610153 | 1.422463 | 0.003526 | 5.515663 | 0.783313 | 0.864946 | 2.522080 |
| EXP020 | 0.661757 | 0.612956 | 1.453868 | 0.002727 | 5.635810 | 0.778625 | 0.973833 | 2.640947 |
| EXP022 | 0.659898 | 0.611687 | 1.416495 | 0.003513 | 5.494913 | 0.782352 | 0.908057 | 2.560522 |
| ExecutedCandidateOracle | 0.743119 | 0.693231 | 0.761339 | 0.000024 | 2.831560 | 0.865193 | 1.002574 | 2.692688 |

Relative to EXP014, EXP022 gains `+0.001644` Accuracy and reduces mean regret
by `0.005967`, recovering `1.94%` of the Accuracy oracle gap and `0.90%` of
the regret gap. Candidate identity mismatch count is zero, so this is a
gate-only raw-utility intervention.

Relative to EXP014, EXP022 changes 1,105 Stay→Move decisions (568 positive,
537 non-positive; mean true U2 `-0.169821`, sum `-187.653`) and 502 Move→Stay
decisions (242 positive, 260 non-positive; mean `-0.540071`, sum `-271.116`).

**Observation.** Raw utility regression improves the canonical trajectory
Accuracy and mean/P90 regret slightly over EXP014, while regression
correlations remain modest and the sign classifier is imperfect.

**Interpretation.** Preserving utility magnitude provides a small but coherent
trajectory benefit over binary BCE gating. The result supports utility-aware
gating as a research direction, but does not establish a deployable final
policy.

**Decision: ACCEPT (research-direction evidence).** No deployment or Test
acceptance is implied; frozen upstream artifacts remain unchanged.

**Next (human review only):** compare this small utility-aware gain with the
warm-start bandit result before authorizing any further method change.
