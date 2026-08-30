# EXP016 — Second-Step Decision Error Decomposition

## Observation

The Val-only analysis covered 13,987 trajectories, including 9,742 frozen-v0
Move episodes. Exact input alignment passed: Stage B and v0 each contained
13,987 episodes, while the Stage D cache and EXP014 second-step predictions
each contained exactly the 9,742-episode v0-Move subset.

| Variant | Accuracy | Macro-F1 | Mean regret | P90 regret | Headroom | Avg moves |
|---|---:|---:|---:|---:|---:|---:|
| EXP014 | 0.658254 | 0.610153 | 1.422463 | 5.515663 | 0.783313 | 0.864946 |
| OracleGate + LearnedCandidate | 0.720026 | 0.670190 | 0.969138 | 3.778069 | 0.842775 | 1.088082 |
| LearnedGate + OracleCandidate | 0.683778 | 0.638848 | 1.235322 | 4.830733 | 0.803908 | 0.864946 |
| Fixed-first Second-Step Oracle | 0.771502 | 0.725081 | 0.586204 | 1.699901 | 0.890887 | 1.088082 |

Relative to EXP014, the gate-only correction recovers 0.061772 Accuracy
(54.55% of the joint Accuracy gap) and reduces mean regret by 0.453324
(54.21% of the joint regret gap). The candidate-only correction recovers
0.025524 Accuracy (22.54%) and reduces mean regret by 0.187140 (22.38%).
The descriptive Accuracy interaction is 0.025953; it is not a causal
decomposition.

On frozen-v0 Move episodes, learned second-step Stay rate is 75.82% versus
the oracle gate's 43.78%. The confusion counts are: learned Stay/oracle Stay
3,578; learned Stay/oracle Move 3,808; learned Move/oracle Stay 687; learned
Move/oracle Move 1,669. Candidate exact hit is 58.99% conditional on oracle
gate Move (5,477 episodes), and 55.90% conditional on learned gate Move
(2,356 episodes).

## Interpretation

The dominant isolated bottleneck is the second-step Stay/Move gate: replacing
only that gate recovers more than twice the Accuracy and mean-regret fraction
of replacing only p2/p3 candidate selection. Candidate ranking remains a
substantial secondary bottleneck, and the positive interaction indicates that
the two errors are not fully additive. The large fixed-first oracle gap shows
that this result does not imply sequential active perception is ineffective;
it localizes most of the current learned-policy loss to second-step decision
quality under the frozen first action.

The 68.93% of frozen-v0 Stay episodes with positive first-step SafeOracle
utility remains an additional first-step exploration ceiling. EXP016 does not
alter or correct those episodes.

## Decision

**INCONCLUSIVE (analysis-only).** EXP016 does not accept a deployable policy
and does not authorize a follow-up training experiment.

## Next

Human review may consider (1) a separately registered second-step gate
intervention, or (2) a candidate-ranking intervention after the gate design is
specified. No intervention is started automatically.

`test_used=false`; no training, Habitat rendering, perception regeneration or
Stage A/B/C-v0, EXP014 or EXP015 modification was performed.
