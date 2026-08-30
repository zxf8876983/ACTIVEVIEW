# Current Task

## EXP016 preparation — second-step decision error decomposition

Stage C-v0, corrected EXP014 and corrected EXP015 remain frozen. EXP016 is an
analysis-only Val decomposition of the second-step gate versus p2/p3 candidate
selection. The frozen Stage C-v0 first decision and Top-1 proposal are retained
for every variant.

## Preparation completed

- Added offline policy primitives for LearnedGate/LearnedCandidate,
  OracleGate/LearnedCandidate, LearnedGate/OracleCandidate and the fixed-first
  oracle.
- Added a Val-only CLI with an explicit parser that rejects every split except
  `val`.
- Added the EXP016 README, analysis record, preparation result and run script.
- Added synthetic unit tests for gate/candidate isolation, negative-U2 edge
  cases, deterministic ties, frozen first-step behavior and Test rejection.
- Corrected EXP016 OracleCandidate ties to match frozen EXP015 cache-order
  `np.argmax` semantics; learned candidates retain geodesic/ID tie-breaking.
- Added fail-closed exact episode-ID alignment for Stage B/v0 and the
  v0-Move-only Stage D cache/EXP014 prediction subset.

## Protocol boundaries

- Real EXP016 Val analysis has **not** been executed.
- Test remains locked and is not accepted or read by the EXP016 entry point.
- No training, Habitat rendering, perception regeneration, ST-GCN retraining,
  Stage A/B/C-v0 modification, EXP014 rerun or EXP015 rerun was performed.

## Status

Prepared for human code review and explicit authorization before the real Val
analysis. Do not start another experiment automatically.
