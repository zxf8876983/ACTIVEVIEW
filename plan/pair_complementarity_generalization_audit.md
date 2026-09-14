# Pair complementarity generalization audit

## Goal

Determine whether PairMeanGreedy gains on reduced12 Moving Val come from
stable viewpoint complementarity or from single-view quality and absolute
viewpoint-ID shortcuts.

## Phases

- [x] Inspect attachment, protocol, frozen caches, and previous sweep.
- [x] Implement Train/Moving-Val-only audit and required metrics.
- [x] Run CUDA audit and verify reproduction gate.
- [x] Review outputs, update project notes, commit only task files, and push.

## Constraints

- Use frozen reduced12 ST-GCN/shared head and normalized MeanLogP.
- Use `current/Stay + Stage-A legal candidate_pool`; no policy Test.
- No recognizer training, perception regeneration, or new model.
- Train labels/evidence may form priors; Val candidate evidence is terminal
  evaluation/oracle only.

## Key questions

1. Does additive single-view quality explain PairMeanGreedy?
2. Does residual pair interaction beat Random B3 by at least 3 pp?
3. How robust are priors to azimuth-ID cyclic shifts and recognizer changes?

## Status

**Completed 2026-09-14** — CUDA audit and protocol gate passed; outputs and
project notes are recorded, with only task-owned files staged for commit.
