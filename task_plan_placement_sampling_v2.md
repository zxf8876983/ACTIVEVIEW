# Task Plan: Furniture-Anchored HM3D Placement Sampling

## Goal
Generate deterministic, furniture-near human placement coordinates for the 21 frozen HM3D-train scenes, using the 14-class raw-val protocol only as provenance, without generating skeletons or observations.

## Phases
- [x] Phase 1: Inspect canonical scene, semantic furniture, and raw-val manifests
- [x] Phase 2: Implement the placement-only CLI
- [x] Phase 3: Compile and run the positions-only generation in Conda Habitat
- [x] Phase 4: Validate outputs and document the stop point

## Decisions Made
- Use eight placements per scene because the detailed placement protocol explicitly specifies `num_placements=8`.
- Reuse the existing semantic center conversion `[x, y, z] -> [x, z, -y]` and Habitat navmesh checks.
- Keep source semantic manifests and raw BABEL/AMASS data read-only.
- Do not generate skeleton, RGB, depth, or policy artifacts in this task.

## Status
**Complete** - all 21 frozen HM3D-train scenes have eight validated positions;
no skeleton or observation generation was started.
