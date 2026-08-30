# EXP015 — Sequential Budget / Oracle Analysis

**Status: PLANNED. No execution has been run.**

## Scientific question

Given the frozen Stage C-v0 proposal mechanism, how much headroom remains after
one real observation, and what recognition/cost trade-off follows from allowing
a second move?

## Methods

The analysis is Val-only and compares NoMove, frozen Stage C-v0 one-step,
EXP014, Fixed-first Second-Step Oracle, CandidateOracle and SafeOracle. The
fixed-first oracle must keep every frozen-v0 first Move/Stay decision unchanged;
only after v0 moves may it choose the best of Stay at s1, p2 and p3 using true
second-step utilities. It is an upper bound on second-step decision quality,
not a deployable policy.

The output includes Accuracy, Macro-F1, mean/median/P90 regret, headroom,
average moves and geodesic cost, initial-Stay ceiling, second-step action
match and the A/B/C trajectory decomposition.

If EXP014 Val output is missing, the command fails clearly. No training, Test,
Habitat rendering or perception rerun is allowed.
