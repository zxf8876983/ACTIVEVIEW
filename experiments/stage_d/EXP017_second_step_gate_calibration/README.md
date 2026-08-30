# EXP017 — Second-Step Gate Calibration Audit

**Status: COMPLETED; Val-only result is REJECT.**

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

## Result

Train selected `tau=-0.08218251913785934`. On Val, the calibrated policy
reduced Accuracy from `0.658254` to `0.650962`, reduced Macro-F1 from
`0.610153` to `0.598102`, increased mean regret from `1.422463` to `1.477153`,
and reduced headroom from `0.783313` to `0.777146`. It changed 2,838
second-step decisions from Stay to Move; candidate identity was unchanged
whenever both policies moved. The scalar-threshold intervention is therefore
rejected as a deployable policy; see `analysis.md`.

## Reproduction command

`run.sh` records the authorized Train-calibrate → freeze artifact → Val-only
evaluation entry point. It has no Test input or Test CLI path.

## Files

- `analysis.md`: preparation record and scientific decision placeholder.
- `result.json`: machine-readable preparation status.
- `run.sh`: explicit Train-calibrate → freeze artifact → Val-evaluate entry
  point.
