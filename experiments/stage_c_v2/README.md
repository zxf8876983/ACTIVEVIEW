# Stage C-v2 Architecture Experiments

This directory contains three independent, Val-only policy experiments.
Stage A, Stage B, Stage C-v0, the 589/197/194 record split, candidate
protocol, perception pipeline and frozen ST-GCN checkpoint remain unchanged.
EXP008--EXP010 have completed their authorized Train-to-Val runs and are
recorded as rejected diagnostic directions; Test remains locked and no Test
evaluation was performed.

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

Each `run.sh` is a reproducible Train-to-Val entry point. Runtime artifacts are
stored below `ACTIVEVIEW_DATA_ROOT/experiments/stage_c_v2/` and are not tracked
in Git. The current skeleton representation is body-yaw canonicalized and
therefore does not preserve explicit body-to-candidate directional alignment;
this limitation was preregistered before these runs and no body-yaw feature was
added.

## Val results

The frozen Stage C-v0 Val baseline was Accuracy 0.649103, Macro-F1 0.598042,
mean regret 1.450498, P90 regret 5.607818, headroom 0.777965 and C2 rate
0.317438.

| Experiment | Accuracy | Macro-F1 | Mean regret | P90 regret | Headroom | C2 rate | Selected epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| EXP008 | 0.644813 | 0.598766 | 1.474656 | 5.633397 | 0.770660 | 0.332952 | 26 |
| EXP009 | 0.647530 | 0.597071 | 1.502271 | 5.819350 | 0.759083 | 0.308572 | 22 |
| EXP010 | 0.651677 | 0.598782 | 1.458965 | 5.660032 | 0.784780 | 0.326875 | 54 |

These Val-only diagnostics did not break the v0 plateau and are retained as
negative evidence for the next predictability-audit phase.
