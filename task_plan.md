# Task Plan: GT human mask + depth root recovery ceiling audit

## Goal
Measure whether a perfect current-frame human mask plus Habitat depth can recover the root well enough to close the D1 NBV gap.

## Phases
- [x] Phase 1: Inspect existing RGB-D audit, dataset loaders, and Habitat render capabilities
- [x] Phase 2: Implement compact GT-mask point-cloud root estimators and D1 evaluator
- [x] Phase 3: Run Train/Moving-Val audit with Habitat CUDA and save reports
- [x] Phase 4: Verify outputs, update project notes, commit and push

## Constraints
- Policy Train and Moving Val only; no Policy Test.
- Current frame 0 only; no candidate RGB/depth, future frames, or new skeleton/perception caches.
- Frozen Yaw8 encoder/shared head and Stage-A legal candidate-only action set.
- Do not submit raw depth, masks, point clouds, large NPZ, or debug images.

## Status
**Completed** - the audit ran on CUDA with eight Habitat workers; compact
reports were generated and are ready for review.
