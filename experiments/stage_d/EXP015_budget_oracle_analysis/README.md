# EXP015 — Sequential Budget / Oracle Analysis

**Status: COMPLETED (Val-only; analysis-only; no Test evaluation).**

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

## Execution record

EXP015 was executed on 2026-08-30 using the completed EXP014 Val predictions
and frozen v0 Val predictions. It performed no training and did not access
Test. The full machine-readable result is `result.json`; the runtime output
is under `ACTIVEVIEW_DATA_ROOT/experiments/stage_d/EXP015_budget_oracle_analysis/`.

Key findings on 13,987 Val episodes:

- Frozen-v0 Stay episodes: 4,245; 2,926 of these (68.93%) would move under
  SafeOracle, with mean missed-move utility 1.732.
- EXP014 second-step action match: 46.73% over 9,742 eligible episodes;
  move-only candidate exact hit: 22.42%.
- Fixed-first Second-Step Oracle: Accuracy 0.771502, Macro-F1 0.725081,
  mean regret 0.586204, P90 regret 1.699901, aggregate headroom capture
  0.890887.

This is an analysis-only upper-bound study; it does not accept a deployable
policy. See `analysis.md` for the controlled interpretation and decision.
