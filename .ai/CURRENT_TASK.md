# Current Task

## EXP017 — second-step gate calibration preparation

Stage C-v0, corrected EXP014, corrected EXP015 and EXP016 remain frozen.
EXP017 is prepared as a no-training audit of whether a Train-selected scalar
Move/Stay threshold can recover the corrected EXP014 second-step gate gap.

The implementation keeps the frozen Stage C-v0 first decision and p1 proposal,
retains EXP014's learned p2/p3 candidate ordering, fits exactly one strict
`gate_score > tau` threshold on Train, writes a frozen calibration artifact,
and only then evaluates Val. The real Train calibration and Val evaluation have
not been executed. Test remains locked.

## Preparation status

- Added `stage_d_gate_calibration.py` with deterministic threshold candidates,
  Train-only selection, gate diagnostics, candidate-identity auditing and
  calibration-artifact validation.
- Added a Val-only-after-calibration CLI that can generate missing frozen
  EXP014 Train predictions by inference only; it never accepts Test.
- Added EXP017 README, analysis placeholder, result status and run script.
- Added focused synthetic tests for strict threshold semantics, Train-only
  fitting, deterministic candidates, identity preservation and artifact guards.

## Protocol boundaries

- No real EXP017 Train calibration or Val evaluation was run.
- No Test data was read.
- No training, Habitat rendering, perception regeneration, ST-GCN retraining,
  Stage A/B/C-v0 modification or EXP014/EXP015/EXP016 rerun was performed.

## Status

EXP017 preparation is **INCONCLUSIVE** pending human code review and explicit
execution authorization. Do not run the supplied `run.sh` automatically.
