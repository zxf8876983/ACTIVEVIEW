# EXP010 — Skeleton Policy Transformer

## Status

COMPLETED — Train-to-Val run finished; pending human scientific review.

## Hypothesis

The frozen action-classification embedding may not be an adequate primary
representation for active viewpoint reasoning. A lightweight policy encoder
that directly models the current estimated skeleton sequence can retain
temporal and joint structure while using only frozen ST-GCN semantic outputs as
context.

## Pre-registered representation limitation

The current skeleton representation is body-yaw canonicalized and therefore
does not preserve explicit body-to-candidate directional alignment.

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

`run.sh` built/loaded the shared current-observation cache, trained on Train
and evaluated Val only. Test was not used.

## Val result

Selected epoch: **54**. Accuracy 0.651677, Macro-F1 0.598782, mean regret
1.458965, median regret 0.004437, P90 regret 5.660032, positive headroom
capture 0.784780, and C2 rate 0.326875. Relative to the frozen v0 Val
baseline, the deltas were Accuracy +0.002574, Macro-F1 +0.000740, mean regret
+0.008467, P90 regret +0.052214, headroom +0.006814, and C2 +0.009437.

Runtime artifacts: `ACTIVEVIEW_DATA_ROOT/experiments/stage_c_v2/EXP010_skeleton_policy_transformer/`.
The body-yaw-canonicalized skeleton limitation above remains in force.
