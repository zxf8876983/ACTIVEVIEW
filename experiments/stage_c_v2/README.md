# Stage C-v2 Architecture Experiments

This directory contains three independent, **planned** Val-only policy
experiments. Stage A, Stage B, Stage C-v0, the 589/197/194 record split,
candidate protocol, perception pipeline and frozen ST-GCN checkpoint remain
unchanged. No experiment in this directory has been trained or evaluated.

The shared cache is built from accepted current-view skeletons and the frozen
ST-GCN. The final ST-GCN block output is captured before global average
pooling (`[B, 256, 8, 17]` for the accepted 30-frame input) and temporally
averaged to `[B, 17, 256]`. The
cache also keeps the current normalized H36M-17 sequence (`[B, 3, 30, 17]`)
and the 19-D current classification context. Candidate inputs contain only
the accepted 11-D geometry; candidate perception and utility remain forbidden.

Experiments:

- `EXP008_joint_aware_current`: mean-pooled joint-aware frozen ST-GCN tokens
  with the existing Set Ranker-style candidate head.
- `EXP009_candidate_conditioned_attention`: candidate geometry queries the
  current frozen joint tokens with one cross-attention layer.
- `EXP010_skeleton_policy_transformer`: a lightweight two-layer Transformer
  encodes the current 30x17 skeleton, followed by candidate-conditioned
  attention.

Each `run.sh` is a reproducible Train-to-Val entry point but is intentionally
not run as part of this preparation task. Test remains locked.
