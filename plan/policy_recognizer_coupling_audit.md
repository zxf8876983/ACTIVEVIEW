# Policy–Recognizer Coupling Audit

## Goal

Determine whether the 52.6984% strict Frame0 result is primarily fixed
viewpoint-prior specialization, recognizer-policy distribution matching, or
instance-conditioned Frame0 information.

## Phases

- [x] Inventory existing selector/head/cache artifacts and protocol gates.
- [x] Implement read-only Train/Moving-Val coupling audit.
- [x] Run CUDA audit and review leakage/protocol checks.
- [x] Update context, commit only task-owned artifacts, and push.

## Constraints

- No new model training, selector retraining, perception generation or Policy
  Test access.
- Strict Frame0 → one legal viewpoint → selected real O1-alone HAR.
- Selector inputs remain current frame-0 DINO/geometry/visibility only.

## Status

**Completed** — implementation, CUDA Val run, context updates and task-owned
commit are complete; the commit is ready to push.
