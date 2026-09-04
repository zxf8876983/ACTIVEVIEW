# Task Plan: Reduced-12 BABEL Protocol and ST-GCN

## Goal
Create an independent 12-class BABEL protocol with diversity-aware caps,
generate skeleton tensors, train a new ST-GCN, and preserve the frozen
selected16 dataset unchanged.

## Phases
- [x] Phase 1: Inspect BABEL/AMASS metadata and existing generation/training APIs
- [x] Phase 2: Implement deterministic diversity-aware selection and manifests
- [x] Phase 3: Generate reduced-12 skeleton datasets and validate split integrity
- [x] Phase 4: Train the new ST-GCN in Conda habitat with CUDA checks
- [x] Phase 5: Record results, run focused validation, and commit source/results

## Fixed protocol
- 12 labels supplied by the user
- Official BABEL Train: maximum 300 samples per class, split 90/10 for ST-GCN
- Official BABEL Val: maximum 100 samples per class, split 60/20/20 for ActiveView
- deterministic seed 42; diversity priority: source, subject, AMASS dataset, duration
- existing selected16 data, manifests, checkpoints and results remain untouched

## Status
**Complete** - manifests, estimated skeleton tensors, independent checkpoint,
and protocol documentation are prepared and validated.
