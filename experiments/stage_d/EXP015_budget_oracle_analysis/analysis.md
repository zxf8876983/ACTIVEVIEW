# EXP015 analysis

## Observation

The corrected Val-only analysis covered 13,987 episodes and consumed the
rebuilt EXP014 cache. Of 4,245 episodes where frozen v0 stayed, SafeOracle
moved in 2,926 cases (68.93%), with mean missed-move utility 1.731704. This
first-step ceiling is unchanged by the azimuth feature correction.

Among 9,742 v0-move episodes, corrected EXP014 matched the fixed-first oracle
action on 46.33% of episodes and had a 17.07% move-only candidate exact hit
rate. Its predicted Stay rate was 75.82% (Move 24.18%), with Stay precision
48.44% and recall 83.89%.

The fixed-first Second-Step Oracle is unchanged because it uses true U2 targets:
Accuracy 0.771502, Macro-F1 0.725081, mean regret 0.586204, P90 regret
1.699901, aggregate positive headroom capture 0.890887.

The corrected EXP014 trajectory decomposition was:

| Group | Episodes | Accuracy | Mean regret |
|---|---:|---:|---:|
| A: v0 Stay | 4,245 | 0.741343 | 1.193631 |
| B: v0 Move, EXP014 Stay | 7,386 | 0.642296 | 1.348446 |
| C: v0 Move, EXP014 Move | 2,356 | 0.558574 | 2.066806 |

## Interpretation

The main structural limit remains the frozen first-step Stay gate: it discards
many cases in which SafeOracle would move. Conditional on reaching the second
step, the corrected learned policy is still far below the fixed-first oracle,
and its exact candidate hit rate is low. The oracle is an upper bound, not a
deployable policy, and this analysis does not justify claims about Test or
unseen-scene generalization.

## Decision

**INCONCLUSIVE.** EXP015 is an analysis-only budget diagnostic. It does not
accept a deployable policy or alter the frozen protocol.

## Next

Human review should decide whether the first-step Stay ceiling warrants a new,
separately registered experiment. No follow-up experiment is started
automatically.

`test_used=false`; Test was not read or used.
