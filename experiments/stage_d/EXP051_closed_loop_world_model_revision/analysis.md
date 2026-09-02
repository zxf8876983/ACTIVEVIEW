# EXP051 analysis

The preregistered gate was met by EXP050 (JOINT_REVISION exceeded the
independent CF_CORRECTNESS_MLP by >0.005 Accuracy without a Macro-F1 drop).
The H=2 run is **blocked** in this campaign because the frozen WM-E checkpoint
requires RGB history inputs, while the approved EXP046 all-view cache stores
recognition states but no frozen RGB embedding for newly visited p2/p3 views.
Producing those embeddings would expand the authorized observation pipeline.
No substitute (zero RGB, true labels, or fixed initial-state predictions) was
used. This is not an EXP051 result and no Test data were read.
