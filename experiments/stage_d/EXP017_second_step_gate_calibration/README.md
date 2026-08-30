# EXP017 — Second-Step Gate Calibration Audit

**Status: PREPARED; real Train calibration and Val evaluation not executed.**

EXP017 is a no-training diagnostic of the corrected frozen EXP014 second-step
policy. Its single intervention is to replace the fixed `tau=0` Move/Stay
decision with one scalar threshold selected on Train only:

```text
gate_score = max(pred_u2_p2, pred_u2_p3)
Move iff gate_score > tau
```

The learned p2/p3 ordering remains the EXP014 deterministic ordering
(predicted utility, geodesic distance, viewpoint ID). The frozen Stage C-v0
first action and p1 proposal are unchanged. True U2 is used only to form the
offline Train calibration label and the non-deployable OracleGate reference;
it is never used to select a deployable candidate.

## Protocol

- Train rows select exactly one threshold by gate balanced accuracy.
- Ties are resolved by greater Move F1, threshold closest to zero, then the
  numerically larger threshold.
- The calibration artifact is written and reloaded before Val is evaluated.
- Val is evaluation-only; Test is rejected and is never read.
- No Stage A/B/C artifact, perception output, Habitat observation or EXP014
  checkpoint is modified.

## Intended command

`run.sh` is provided for a later, explicitly authorized execution. This
preparation task does not invoke it.

## Files

- `analysis.md`: preparation record and scientific decision placeholder.
- `result.json`: machine-readable preparation status.
- `run.sh`: explicit Train-calibrate → freeze artifact → Val-evaluate entry
  point.
