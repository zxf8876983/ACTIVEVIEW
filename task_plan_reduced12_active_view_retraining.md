# Task Plan: Reduced12 eight-placement ActiveView retraining

## Goal
Train the formal Recognition-aware WM-E + Multi-positive JR + closed-loop H2 protocol on the new reduced12 eight-placement Train/Val assets without reading policy Test or overwriting older experiments.

## Phases
- [ ] Phase 1: Audit assets, schemas, and compatible entrypoints
- [ ] Phase 2: Build Train/Val policy artifacts and validate counts
- [ ] Phase 3: Train formal initial policy, WM-E, and Multi-positive JR on Train
- [ ] Phase 4: Run Val benchmark, H0 oracles, WM diagnostics, and H2 evaluation
- [ ] Phase 5: Write result/analysis artifacts, verify scope, commit and push

## Frozen constraints
- Taxonomy is the current 12-label reduced12 protocol.
- Do not read policy Test or use its files as inputs.
- Do not overwrite reduced14/old16 artifacts or source raw data.
- Use `/home/zxf/anaconda3/envs/habitat/bin/python` and CUDA when available.

## Status
**Phases 1–4 completed** — Train/Val assets, Stage A–D caches, frozen initial policy, formal WM-E/JR training, and Val benchmark are complete. Phase 5 is limited to scope verification, documentation, commit, and push.
