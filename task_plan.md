# Task Plan: Known-map + current-depth movement-aware NBV audit

## Goal
Measure whether known Habitat navmesh path cost and current Frame-0 depth
occupancy/dynamic-human risk can reduce movement while preserving HAR quality.

## Phases
- [x] Validate reduced12 Yaw8 assets, CUDA and Stage-A candidate protocol
- [x] Implement compact current-depth occupancy/path-risk cache (8 workers)
- [x] Run Moving-Val evaluation and Train record-holdout lambda selection
- [x] Write required metrics/analysis and verify outputs

## Constraints
- Policy Train and Moving Val only; no Policy Test.
- Current frame 0 only; no candidate RGB/depth, future frames, or new skeleton/perception caches.
- Frozen Yaw8 encoder/shared head and Stage-A legal candidate-only action set.
- Do not submit raw depth, point clouds, large caches, or debug images.

## Status
**Completed** - 10,080 Moving-Val contexts and 68,702 legal candidates were
evaluated with the frozen Yaw8 recognizer. Current depth was rendered only at
the current Frame-0 viewpoint; raw depth and point clouds were not persisted.
RGBGlobal-Visibility + MapPath selected lambda=0.10 on a Policy-Train record
holdout, reducing mean path by 4.46% with a 0.248pp Accuracy drop, below the
20% movement gate. Adding current depth switched 5.34% of selections and
reduced depth-risk rate by 5.45% with 5.15% relative clearance improvement,
but did not meet the strict depth KEEP criteria. Policy Test was not read.
