# EXP009 — Candidate-Conditioned Joint Attention

## Status

PLANNED — code/config prepared; no training or evaluation has been run.

## Hypothesis

Viewpoint utility may require candidate-specific access to different parts of
the current skeleton. A candidate geometry query over frozen joint-aware
tokens should provide that conditioning while retaining permutation-equivariant
candidate-set interaction.

## Single change

Each accepted 11-D candidate geometry is mapped to a 128-D query and performs
one four-head cross-attention operation over the current `[17, 256]` frozen
ST-GCN joint tokens. The resulting candidate-conditioned representation is
scored by the existing Set Ranker-style interaction head. No candidate
perception is read.

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
