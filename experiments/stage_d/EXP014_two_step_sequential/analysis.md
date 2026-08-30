# EXP014 analysis

## Observation

On 13,987 Val episodes, EXP014 reached Accuracy 0.664331 and Macro-F1
0.615151, compared with frozen Stage C-v0 values 0.649103 and 0.598042.
Mean regret decreased from 1.450498 to 1.397287 (3.67% relative reduction),
and P90 regret decreased from 5.607818 to 5.403128 (3.65%). Aggregate positive
headroom capture changed from 0.777965 to 0.783344. The policy averaged 0.916
moves and 2.562 m trajectory cost, versus 0.697 moves and 2.201 m for v0.

## Interpretation

The single visited intermediate observation provides a measurable but modest
Val improvement in recognition and regret. The improvement is below the
pre-registered strong-success thresholds, and the extra movement budget is
non-trivial. This does not establish that the sequential policy is ready for
deployment; it only shows that the approved two-step construction is viable for
further human review.

## Decision

**INCONCLUSIVE.** The explicit reject condition (both Accuracy below 0.66 and
mean-regret reduction below 5%) was not satisfied, while the strong-success
criteria were not satisfied either.

## Next

Human review should decide whether the measured gain justifies the added
movement cost and whether to retain the sequential direction. No follow-up
experiment is started automatically.

`test_used=false`; Test was not read or used for selection.
