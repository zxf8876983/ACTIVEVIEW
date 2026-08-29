# EXP008 — Joint-Aware Current Representation

## Status

COMPLETED — diagnostic direction rejected after Train-to-Val review.

## Hypothesis

The globally pooled 275-D ST-GCN representation discards joint-level evidence
that is useful for viewpoint utility prediction. Mean-pooled final-block joint
tokens should preserve this information without introducing a new temporal
model or changing the candidate interaction head.

## Pre-registered representation limitation

The current skeleton representation is body-yaw canonicalized and therefore
does not preserve explicit body-to-candidate directional alignment.

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

`run.sh` built/loaded the shared current-observation cache, trained on Train
and evaluated Val only. Test was not used.

## Val result

Selected epoch: **26**. Accuracy 0.644813, Macro-F1 0.598766, mean regret
1.474656, median regret 0.006556, P90 regret 5.633397, positive headroom
capture 0.770660, and C2 rate 0.332952. Relative to the frozen v0 Val
baseline, the deltas were Accuracy -0.004290, Macro-F1 +0.000724, mean regret
+0.024159, P90 regret +0.025580, headroom -0.007305, and C2 +0.015514.

Decision: **REJECT** as a v2 diagnostic direction; the primary regret metrics
did not improve over the frozen v0 Val baseline. Runtime artifacts:
`ACTIVEVIEW_DATA_ROOT/experiments/stage_c_v2/EXP008_joint_aware_current/`.
The body-yaw-canonicalized skeleton limitation above remains in force.
