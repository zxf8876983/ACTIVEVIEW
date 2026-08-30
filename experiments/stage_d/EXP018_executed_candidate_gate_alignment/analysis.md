# EXP018 analysis

The Val-only analysis is executed by `run.sh` after implementation. This
file records the frozen-protocol interpretation of the generated result.

## Observation

See `result.json` for the four-policy table, target-alignment counts and the
decomposition of EXP017's Stay→Move episodes. The analysis uses 13,987 Val
episodes, of which the frozen v0 first decision moves on the eligible subset.

## Interpretation

The key comparison is between AnyPositiveOracleGate and
ExecutedCandidateOracleGate, while the learned candidate identity remains
the corrected EXP014 ranking. A lower executed-candidate oracle indicates
that part of apparent gate headroom is not executable under the current
ranking. The EXP017 changed-gate breakdown directly checks whether its extra
moves were useful for the candidate it would actually execute.

## Decision

**INCONCLUSIVE** — this is an offline diagnostic, not a deployable policy
acceptance. It does not authorize EXP019 or any model change.

## Validity

`test_used=false`, `training_performed=false`, no perception regeneration,
no Habitat rendering and no ST-GCN retraining. Stage A/B/C-v0, EXP014,
EXP015, EXP016 and EXP017 artifacts are unchanged.
