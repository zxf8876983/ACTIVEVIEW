# EXP032 Utility provenance

The frozen Stage-B utility builder computes, for current view `s0` and candidate
view `p`,

`U(p) = log p_STGCN(y_true | p) - log p_STGCN(y_true | s0)`.

The implicit Stay action is defined as `U_stay = 0`. Candidate utilities are
relative-to-Stay improvements; the oracle selects the largest candidate utility,
with the existing deterministic geodesic/viewpoint tie rule, and the safe oracle
stays when the maximum is non-positive. The target directly uses frozen ST-GCN
true-label log probability (equivalently CE/probability), but does not use pose
error. `margin_1` is the difference between the largest and second-largest value
in `[0, U(p2), U(p3)]`. Thus `future_recognition_quality_is_part_of_utility_definition=true`.

EXP032-B is consequently a privileged decomposition/sanity upper bound, not an
independent causal claim.
