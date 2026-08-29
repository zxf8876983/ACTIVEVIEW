# EXP009 — Candidate-Conditioned Joint Attention

## Status

COMPLETED — Train-to-Val run finished; pending human scientific review.

## Hypothesis

Viewpoint utility may require candidate-specific access to different parts of
the current skeleton. A candidate geometry query over frozen joint-aware
tokens should provide that conditioning while retaining permutation-equivariant
candidate-set interaction.

## Pre-registered representation limitation

The current skeleton representation is body-yaw canonicalized and therefore
does not preserve explicit body-to-candidate directional alignment.

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

`run.sh` built/loaded the shared current-observation cache, trained on Train
and evaluated Val only. Test was not used.

## Val result

Selected epoch: **22**. Accuracy 0.647530, Macro-F1 0.597071, mean regret
1.502271, median regret 0.005624, P90 regret 5.819350, positive headroom
capture 0.759083, and C2 rate 0.308572. Relative to the frozen v0 Val
baseline, the deltas were Accuracy -0.001573, Macro-F1 -0.000971, mean regret
+0.051773, P90 regret +0.211532, headroom -0.018882, and C2 -0.008865.

Runtime artifacts: `ACTIVEVIEW_DATA_ROOT/experiments/stage_c_v2/EXP009_candidate_conditioned_attention/`.
The body-yaw-canonicalized skeleton limitation above remains in force.
