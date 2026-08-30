# EXP014 analysis

## Observation

The corrected Val-only rerun covered 13,987 episodes. The Stage D cache was
rebuilt using the Stage A-compatible radial relative azimuth from the existing
`semantic-region-v2` metadata; no perception or upstream artifact changed.

| Metric | Frozen Stage C-v0 | EXP014 corrected | Delta |
|---|---:|---:|---:|
| Accuracy | 0.649103 | 0.658254 | +0.009151 |
| Macro-F1 | 0.598042 | 0.610153 | +0.012111 |
| Mean regret | 1.450498 | 1.422463 | −0.028035 |
| Median regret | 0.005143 | 0.003526 | −0.001618 |
| P90 regret | 5.607818 | 5.515663 | −0.092155 |
| Aggregate positive headroom capture | 0.777965 | 0.783313 | +0.005347 |

Mean regret improved by 1.93% relative to frozen v0 and P90 regret by 1.64%.
The selected checkpoint was epoch 11. The policy averaged 0.865 moves per
episode and 2.522 m trajectory geodesic cost (v0: 0.697 moves and 2.201 m).

For traceability, the pre-fix runtime is archived under
`ACTIVEVIEW_DATA_ROOT/experiments/stage_d/EXP014_two_step_sequential_pre_geometry_fix/`.
The corrected run is the only result used for the decision below.

## Interpretation

One visited intermediate observation still provides a positive but small Val
gain over the frozen one-shot policy. The gain is below the preregistered
strong-success thresholds (Accuracy ≥0.68 and ≥10% mean-regret reduction).
Under the recorded explicit reject condition (Accuracy <0.66 and mean-regret
reduction <5%), both conditions are met. The corrected radial-azimuth semantics
also reduce the apparent gain relative to the invalid pre-fix run, confirming
that the earlier result cannot be retained as evidence.

EXP014 remains a two-step offline Val study; it does not establish unseen-scene
generalization or readiness for online deployment.

## Decision

**REJECT.** The corrected run does not meet the strong-success criteria and
meets the recorded reject thresholds. This rejects the current EXP014 policy
configuration, not the general possibility of sequential active perception.

## Next

Human review may decide whether to retain the sequential direction as a new,
explicitly registered experiment. No follow-up experiment is started
automatically.

`test_used=false`; Test was not read or used for selection.
