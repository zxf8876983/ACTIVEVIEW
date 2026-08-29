# Current Task

## Current experiment

EXP002 — hard-record-aware sampling

Status: **PLANNED**; human review is required before training.

## Goal

Test whether Train-only difficult motion records are underrepresented during
Stage C training.

## Evidence

The frozen EXP001 Val analysis found that the worst 10% of motion records
accounted for 37.43% of catastrophic failures (approximately 3.74×
concentration). A frozen Stage C-v0 Train inference identified 118 hard records
among 589.

## Single Change

Replace record-balanced sampling with a hard-record-aware sampler. Hard records
sample 32 Episodes/epoch; normal records sample 12. The base loss is restored
with `lambda_gap=0`.

## Frozen

Architecture, features, Stage A/B/C-v0 artifacts, loss, optimizer, scheduler,
split, ST-GCN, candidate protocol, decision rule and metrics.

## Primary

Val P90 regret; target at least 5% relative reduction from 5.6078176767.

## Next

Human review before training.

## Do not

- run EXP002;
- run Test;
- change geometry or loss;
- create EXP003.
