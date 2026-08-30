# EXP018 analysis

## Observation

The frozen Val universe contains 13,987 episodes and 9,742 v0-Move episodes.
The four-policy metrics are:

| Variant | Accuracy | Macro-F1 | Mean regret | Median regret | P90 regret | Headroom | Avg moves | Mean geodesic (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP014, learned gate | 0.658254 | 0.610153 | 1.422463 | 0.003526 | 5.515663 | 0.783313 | 0.864946 | 2.522080 |
| EXP017, calibrated gate | 0.650962 | 0.598102 | 1.477153 | 0.004613 | 5.605605 | 0.777146 | 1.067849 | 2.793787 |
| Any-positive oracle gate + learned candidate | 0.720026 | 0.670190 | 0.969138 | 0.000029 | 3.778069 | 0.842775 | 1.088082 | 2.832175 |
| Executed-candidate oracle gate + learned candidate | 0.743119 | 0.693231 | 0.761339 | 0.000024 | 2.831560 | 0.865193 | 1.002574 | 2.692688 |

Among v0-Move episodes, `y_any` is positive for 5,477 episodes and
`y_exec` is positive for 4,281. There are 1,196 ranking-induced gate
mismatches (`y_any=Move`, `y_exec=Stay`), or 21.84% of any-positive episodes.
The impossible `y_any=Stay`, `y_exec=Move` combination occurred zero times.

EXP017's 2,838 additional Stay→Move decisions contain 1,341 executed-positive
and 1,497 executed-nonpositive cases. Their executed true-U2 mean is
`-0.269540` and median is `-0.0000846`.

## Interpretation

The executed-candidate oracle is higher than the any-positive oracle
(0.743119 vs 0.720026 Accuracy; 0.761339 vs 0.969138 mean regret) because
the any-positive gate sometimes forces a move even when the frozen learned
candidate is harmful. Thus, the previous any-positive gate ceiling was not
the executable ceiling under the fixed candidate ranking. The executed gate
leaves a larger, ranking-compatible diagnostic headroom than EXP014.

More than half of EXP017's extra moves (1,497/2,838 = 52.75%) were
non-positive for the candidate that EXP014 would execute, explaining why the
Train-calibrated threshold worsened trajectory metrics despite improving
any-positive gate recall. This supports an executed-candidate-aligned gate
target as a future hypothesis; it does not establish a deployable policy or
separate the effects of a future learned gate from candidate ranking.

The >1 executable fraction of any-positive headroom (1.373843 Accuracy and
1.458389 regret reduction fractions) is descriptive, not a probability: the
any-positive oracle itself includes harmful forced moves under the frozen
candidate ranking.

## Decision

**INCONCLUSIVE** — EXP018 is an offline diagnostic and no policy is accepted
for deployment. It does not authorize EXP019 or any model change.

## Validity

`test_used=false`, `training_performed=false`, no perception regeneration,
no Habitat rendering and no ST-GCN retraining. Stage A/B/C-v0, EXP014,
EXP015, EXP016 and EXP017 artifacts are unchanged.
