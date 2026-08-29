# EXP008 — Joint-Aware Current Representation

## Status

PLANNED — code/config prepared; no training or evaluation has been run.

## Hypothesis

The globally pooled 275-D ST-GCN representation discards joint-level evidence
that is useful for viewpoint utility prediction. Mean-pooled final-block joint
tokens should preserve this information without introducing a new temporal
model or changing the candidate interaction head.

## Single change

Replace the 256-D globally pooled ST-GCN feature with frozen final-block
joint-aware tokens (`[17, 256]`), projected to 128-D per joint and mean-pooled.
The 19-D current log-probability/entropy/margin/confidence context is retained.
The candidate head remains Set Ranker-style with the accepted 11-D geometry.

## Frozen

Stage A/B/C-v0, ST-GCN weights, utility targets, record-balanced sampler, loss,
optimizer, scheduler, split, candidate pool and decision rule.

## Training contract

Seed 42, 100 epochs maximum, patience 10, batch 128, AdamW (1e-3,
weight-decay 1e-4), SmoothL1 plus stay-inclusive listwise CE. Checkpoint
selection is Val Macro-F1. Test is not used.

## Run

`run.sh` builds/loads the shared current-observation cache, trains on Train and
evaluates Val only. It is not run during this preparation task.
