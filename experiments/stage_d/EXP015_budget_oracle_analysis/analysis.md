# EXP015 analysis

## Observation

The Val-only analysis covered 13,987 episodes. Of 4,245 episodes where frozen
v0 stayed, SafeOracle moved in 2,926 cases (68.93%), with mean missed-move
utility 1.731704. Among 9,742 v0-move episodes, EXP014 matched the fixed-first
oracle action 46.73% of the time and had 22.42% move-only candidate exact hit.
The fixed-first Second-Step Oracle reached Accuracy 0.771502, Macro-F1
0.725081, mean regret 0.586204 and P90 regret 1.699901.

## Interpretation

The analysis shows substantial budget headroom after the first decision. It
also separates two limits: the frozen first-stage Stay gate discards many
potential moves, while the learned second-step decision remains imperfect.
The fixed-first oracle is an upper bound, not a deployable policy, and does not
justify claims about Test or unseen-scene generalization.

## Decision

**INCONCLUSIVE.** EXP015 is an analysis-only diagnostic and does not accept a
new policy or alter the frozen protocol.

## Next

Human review should decide whether the first-stage Stay ceiling warrants a
separate approved experiment. No such experiment is started automatically.

`test_used=false`; Test was not read or used.
