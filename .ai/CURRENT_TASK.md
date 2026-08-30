# Current Task

## EXP016 — second-step decision error decomposition completed

Stage C-v0, corrected EXP014 and corrected EXP015 remain frozen. EXP016 is an
analysis-only Val decomposition of the second-step gate versus p2/p3 candidate
selection. The frozen Stage C-v0 first decision and Top-1 proposal are retained
for every variant.

## Completed analysis

- Added offline policy primitives for LearnedGate/LearnedCandidate,
  OracleGate/LearnedCandidate, LearnedGate/OracleCandidate and the fixed-first
  oracle.
- Added a Val-only CLI with an explicit parser that rejects every split except
  `val`.
- Ran the authorized Val-only analysis over 13,987 trajectories (9,742
  frozen-v0 Move episodes).
- Corrected EXP016 OracleCandidate ties to match frozen EXP015 cache-order
  `np.argmax` semantics; learned candidates retain geodesic/ID tie-breaking.
- Enforced fail-closed exact episode-ID alignment for Stage B/v0 and the
  v0-Move-only Stage D cache/EXP014 prediction subset.
- Stored compact results in the EXP016 experiment record and the full runtime
  result under `ACTIVEVIEW_DATA_ROOT`.

## Protocol boundaries

- Real EXP016 Val analysis has been executed once under the authorized Val-only
  protocol.
- Test remains locked and is not accepted or read by the EXP016 entry point.
- No training, Habitat rendering, perception regeneration, ST-GCN retraining,
  Stage A/B/C-v0 modification, EXP014 rerun or EXP015 rerun was performed.

## Status

Analysis-only result is ready for human scientific review. EXP016 is
**INCONCLUSIVE** and does not authorize a follow-up training experiment.
Do not read Test or start another experiment automatically.
