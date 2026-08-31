# EXP030 — Fast Candidate-Conditioned Visibility / Utility Pilot

Val-only diagnostic of candidate-conditioned visibility. Method A/A0 uses
only depth endpoints observed from visited `s0` and `s1`; Method B regresses
candidate utility; Method C audits p2/p3 ranking; Method D is a separate
full-scene Habitat raycast upper bound. Human anchors are privileged diagnostic
geometry and are never a learned-policy input. No candidate observation,
Test split or trajectory rollout is used.

Runtime feature JSONL and Habitat artifacts remain outside Git under
`ACTIVEVIEW_DATA_ROOT/experiments/stage_d/EXP030_candidate_visibility_pilot`.
