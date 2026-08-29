# EXP010 — Skeleton Policy Transformer

## Status

PLANNED — code/config prepared; no training or evaluation has been run.

## Hypothesis

The frozen action-classification embedding may not be an adequate primary
representation for active viewpoint reasoning. A lightweight policy encoder
that directly models the current estimated skeleton sequence can retain
temporal and joint structure while using only frozen ST-GCN semantic outputs as
context.

## Single change

Encode the current normalized estimated skeleton (`[3, 30, 17]`) as 510
joint-time tokens with learned joint/time embeddings and a two-layer,
four-head Transformer encoder. Each candidate geometry then queries these
tokens with one cross-attention layer. Only the 19-D frozen ST-GCN
log-probability/entropy/margin/confidence context is appended; the pooled 256-D
ST-GCN feature is not used as the policy representation.

## Frozen

Stage A/B/C-v0, frozen ST-GCN outputs and checkpoint, utility targets,
record-balanced sampler, loss, optimizer, scheduler, split, candidate pool and
decision rule. No independent Move/Stay gate is introduced.

## Training contract

Seed 42, 100 epochs maximum, patience 10, batch 64 (the only permitted change
from the common batch size because of the 510-token sequence), AdamW (1e-3,
weight-decay 1e-4), SmoothL1 plus stay-inclusive listwise CE. Checkpoint
selection is Val Macro-F1. Test is not used.

## Run

`run.sh` builds/loads the shared current-observation cache, trains on Train and
evaluates Val only. It is not run during this preparation task.
