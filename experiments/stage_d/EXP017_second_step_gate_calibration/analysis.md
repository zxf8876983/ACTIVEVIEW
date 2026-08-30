# EXP017 — Second-Step Gate Calibration Audit

## Observation

EXP017 used the frozen corrected EXP014 ranker without retraining. A single
scalar threshold was selected on 29,133 Train second-step episodes by gate
balanced accuracy. The selected threshold was:

```text
tau = -0.08218251913785934
```

The authorized Val evaluation contains 13,987 episodes, including 9,742
episodes for which frozen Stage C-v0 moved to `p1`.

| Variant | Accuracy | Macro-F1 | Mean regret | Median regret | P90 regret | Headroom | Avg moves |
|---|---:|---:|---:|---:|---:|---:|---:|
| EXP014, tau=0 | 0.658254 | 0.610153 | 1.422463 | 0.003526 | 5.515663 | 0.783313 | 0.864946 |
| EXP017, Train-calibrated tau | 0.650962 | 0.598102 | 1.477153 | 0.004613 | 5.605605 | 0.777146 | 1.067849 |
| OracleGate + LearnedCandidate | 0.720026 | 0.670190 | 0.969138 | 0.000029 | 3.778069 | 0.842775 | 1.088082 |

Relative to EXP014, EXP017 changes 2,838 episodes from Stay to Move and zero
episodes from Move to Stay. The learned candidate identity is unchanged in
all 2,356 episodes where both policies Move. The changed episodes have mean
selected-utility change `-0.269540`; 1,493 have increased regret and 1,341
have reduced regret.

The gate-only diagnostic does improve on Val: balanced accuracy increases from
0.571825 to 0.609873 and Move recall from 0.304729 to 0.629359. Move
precision decreases from 0.708404 to 0.663650, while Stay recall decreases
from 0.838921 to 0.590387. The frozen gate-score ROC-AUC is 0.656316 and
PR-AUC is 0.682661.

The EXP014 and OracleGate + LearnedCandidate reference checks both match their
frozen values within `1e-5`.

## Interpretation

The intervention was isolated to the second-step gate threshold: candidate
ranking, first-step action, p1 proposal, utility targets and perception were
unchanged. Therefore the candidate-identity audit rules out a ranking change
as the source of the comparison.

Train calibration correctly moves the operating point toward higher Move
recall, but this does not transfer into better trajectory decisions. The
additional 2,838 moves include enough harmful or weak moves to lower
recognition and headroom and increase regret. The negative accuracy and regret
gap-recovery fractions (`-11.81%` and `-12.06%`) show that a single global
threshold is not a reliable solution for the current score calibration.

This supports the controlled conclusion that EXP014's gate bottleneck is not
explained by the zero threshold alone. It does not establish that a separately
trained gate or a richer state representation will succeed; those remain
future hypotheses.

## Decision

**REJECT** — do not retain the Train-calibrated global threshold as the Stage D
policy. EXP017 is a diagnostic intervention, not a deployable policy.

## Next

1. Human review may consider a separately structured second-step gate that is
   evaluated under the same frozen first-step protocol.
2. Alternatively, review whether current-state representation is sufficient
   before authorizing another controlled experiment.

No follow-up experiment was started automatically. Test was not read or used.
