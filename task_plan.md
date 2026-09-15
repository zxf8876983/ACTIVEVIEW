# Task Plan: Frame0 true-facing confidence prior audit

## Goal
Audit a true-body-facing, frame-0 YOLO confidence angle prior using only existing Train/internal-Val and Moving-Val artifacts.

## Phases
- [x] Phase 1: Confirm protocol, schemas, and CUDA runtime
- [x] Phase 2: Implement angle recovery, confidence-schema audit, and selectors
- [x] Phase 3: Run internal prior and Moving-Val diagnostics
- [x] Phase 4: Verify outputs, document limitations, commit and push

## Key Questions
1. Is the frame-0 confidence schema available in the existing archives?
2. Does the true-facing angle convention differ from the old placement-yaw convention?
3. If exact confidence is available, does its Train-internal prior generalize to Moving Val?

## Decisions Made
- Use AMASS/BABEL frame-0 root transform composed with placement scene yaw for body facing.
- Enforce frame-0 keypoint confidence when the archive schema supports `(30,17)` confidence.
- If eight-placement policy archives expose only sequence-level `(32,)` confidence, report it as a non-frame-0 proxy and never use it to claim an exact Moving frame-0 confidence landscape.

## Errors Encountered
- Existing policy archives store only per-view sequence-mean confidence `(32,)`; exact Moving candidate frame-0 confidence is unavailable without forbidden RGB/YOLO regeneration.

## Status
**Completed** - CUDA run, artifacts verified, task ready to commit.
