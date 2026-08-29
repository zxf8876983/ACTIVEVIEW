# Current Task

## Status

Repository simplified for research-first development.

## Current experiment

`EXP001 — utility-gap-aware ranking`

Status: **PLANNED**; human review is required before running it.

## Goal

Test whether the gap-aware ranking term reduces harmful large-gap Val regret.

## Single change

Add `L_gap` to the existing Set Ranker loss. No architecture or input change.

## Frozen

Stage A/B/C-v0 artifacts, features, ST-GCN, Set Ranker architecture, sampler,
candidate protocol and record split.

## Parameters

`lambda_gap=1.0`, `tau_gap=1.0`, `max_gap_weight=10.0`; seed 42; 100 epochs;
batch size 128; AdamW learning rate 0.001 and weight decay 0.0001.

## Val baseline

Val Accuracy 0.6491; Macro-F1 0.5980; mean regret 1.4505; p90 regret 5.6078;
positive headroom capture 0.7780; large-gap mean regret 2.4493; C2 rate 0.3174.

## Next action

Human review, then run `experiments/stage_c_v1/EXP001_gap_aware_ranking/run.sh`.

## Do not

- run Test;
- change frozen methods or data;
- create EXP002;
- start another experiment automatically.
