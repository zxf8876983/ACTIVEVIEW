# Adaptive Head × Frame0 NBV Combination Audit

## Goal

Test whether the historical strict Frame0 RGBGlobal+Geometry+VisibilityAux
selector combines additively with frozen-encoder adaptive recognition heads.

## Phases

- [x] Inventory matched Stage-D caches, historical selector and adaptive-head checkpoints.
- [ ] Implement balanced legal-candidate head and strict Frame0 selector comparison.
- [ ] Run CUDA Train/Moving-Val audit and verify leakage/protocol gates.
- [ ] Review reports, update project context, commit only task-owned files, and push.

## Constraints

- Train/Moving-Val only; never read Policy Test.
- Frame0 current RGB/DINO global feature + legal geometry + historical VisibilityAux are the only selector inputs.
- Candidate recognizer evidence is used for Train targets and Val terminal evaluation/oracle diagnostics only.
- ST-GCN encoder and formal methods remain frozen; new balanced head/checkpoints stay in external runtime.

## Key questions

1. Is the historical adaptive head only specialized to s1, or does it generalize to legal candidates?
2. Does adaptive recognition change candidate utility rankings and strict Frame0 O1-alone performance?
3. Are head and NBV gains additive enough to keep the combination?

## Status

**Currently in implementation** — matched caches, checkpoints and exact Frame0
selector helpers have been identified; no Test artifacts are in scope.
