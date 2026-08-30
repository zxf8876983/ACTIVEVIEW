# EXP017 preparation analysis

## Observation

The implementation and synthetic tests for a Train-only scalar gate
calibration audit are prepared. No real EXP017 Train calibration or Val
evaluation has been executed, so no performance observation is available.

## Interpretation

The experiment is designed to distinguish a miscalibrated zero threshold from
insufficient Move-vs-Stay score discrimination while holding candidate ranking
and the frozen first-step protocol constant. This document intentionally does
not infer either explanation before the authorized run.

## Decision

**INCONCLUSIVE** — preparation only; human review and execution authorization
are required.

## Next

1. Review the implementation and unit-test results.
2. If authorized, run the supplied Val-only-after-Train-calibration command
   once and review the resulting Train calibration artifact before interpreting
   Val metrics.

`test_used=false`, `training_performed=false`, and all upstream perception and
Habitat-generation flags remain false for this preparation record.
*** Add File: experiments/stage_d/EXP017_second_step_gate_calibration/result.json
{
  "experiment_id": "EXP017",
  "experiment_name": "second_step_gate_calibration",
  "status": "PREPARED",
  "split": "val",
  "test_used": false,
  "training_performed": false,
  "perception_regenerated": false,
  "habitat_rendering_performed": false,
  "stgcn_retrained": false,
  "real_train_calibration_executed": false,
  "real_val_evaluation_executed": false,
  "protocol": {
    "frozen_first_step": true,
    "frozen_exp014_candidate_ranking": true,
    "threshold_fit_split": "train",
    "threshold_application_split": "val",
    "decision": "Move iff max(pred_u2_p2, pred_u2_p3) > tau",
    "test_split_accepted": false
  },
  "metrics": null,
  "provenance": {
    "runtime_artifacts": "not generated; execution not authorized"
  }
}
