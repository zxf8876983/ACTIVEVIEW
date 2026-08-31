# EXP030 — Fast Candidate-Conditioned Visibility / Utility Pilot

## Protocol

- Train/Val only: 29,133 / 9,742 eligible Stage-D episodes.
- Only visited `s0`/`s1` depth was rendered (pixel stride 8); candidate views
  were never rendered or loaded.
- Human anchors are privileged frame-15 kinematic geometry (`17` anchors).
- Method D is a separate full-scene Habitat raycast upper bound and is not a
  legal observed-only feature.

## Results

| Method | Val result |
|---|---|
| A0 observed-only analytic winner | 0.509744 winner accuracy on 5,234 oracle-Move episodes |
| A0 visibility/utility correlation | Pearson 0.026681; Spearman 0.009913 |
| B observed-only utility regression | MAE 3.034123; RMSE 4.606810; Pearson 0.420523; Spearman 0.299330 |
| B episode action | 3-way 0.455759; binary 0.566824; candidate hit 0.582401; selected utility 0.042486 |
| C observed-only ranking | winner 0.559610; balanced accuracy 0.547756 |
| D full-scene upper bound | winner 0.559037; Pearson 0.367157; Spearman 0.193020 |

Method B selected actions: Stay 5,817, p2 2,314, p3 1,611; harmful moves
1,858 and missed beneficial moves 2,886. Margin-conditioned B 3-way accuracy
was 0.506165 (margin ≥0.25), 0.519982 (≥0.5), 0.537294 (≥1.0), and 0.571125
(≥2.0). A0 winner accuracy rose to 0.580786 at margin ≥2.0.

## Decision

`CASE_C`: observed-only visibility and the privileged full-scene visibility
diagnostic are both weak candidate predictors in this pilot. This does not
support escalating generic semantic BEV or a deployable visibility branch;
the next investigation should prioritize self-occlusion, pose/viewpoint
uncertainty, or body-orientation representation.

## Validity

`test_used=false`; no trajectory rollout, perception regeneration, YOLO,
VideoPose3D, or ST-GCN retraining. `future_candidate_rgb_used=false`,
`future_candidate_depth_used=false`, `future_candidate_semantic_used=false`,
and `future_candidate_skeleton_used=false`. Method D alone uses complete scene
geometry and is explicitly marked as an oracle upper bound.
