# Task Plan: Train-internal-Val Relative View Quality Prior

## Goal
Build and evaluate a frozen Yaw8Fair relative-view quality prior using only a raw-train-derived internal validation split, then test it once on Moving Val without reading Policy Test.

## Phases
- [x] Phase 1: Confirm assets, split, angle convention and baseline loaders
- [x] Phase 2: Implement internal landscape, frozen priors, fusion and diagnostics
- [x] Phase 3: Run CUDA evaluation and inspect generated artifacts
- [ ] Phase 4: Update project state, commit and push

## Key Questions
1. Does a relative body-view angle ranking learned from raw-train internal validation generalize to Moving Val?
2. Does the relative prior beat StaticPrior or improve RGBGlobal-Visibility when fused?
3. Is angle preference action-dependent or stable across actions?

## Decisions Made
- Use the existing raw-train-derived Yaw8 ST-GCN validation observations as the internal validation population (245 source records × 8 yaw variants).
- Reconstruct the frozen Yaw8 camera azimuth from the recorded generation rule and `babel_sid`; do not inspect raw-val or Moving Val while constructing the angle prior.
- Use the existing Yaw8Fair shared-head candidate cache and frozen Yaw8 checkpoint for Moving Val diagnostics only.

## Errors Encountered
- The Yaw8 checkpoint was stored under `checkpoints/stgcn_reduced12_yaw8_v1/best.pt`, not beside the dataset arrays; the script now resolves that canonical checkpoint path.

## Status
**Ready for Phase 4** - CUDA evaluation completed and all required JSON/Markdown artifacts validated.
