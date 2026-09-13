# O0-conditioned complementary second-view sweep — completed 2026-09-14

Completed the Train/Val-only overnight reduced12 audit for selecting exactly
one additional legal second viewpoint after a complete current-view (O0)
observation. Policy Test was not read; no RGB, skeleton, DINO, recognizer, or
formal checkpoint was changed.

The formal action set was current/Stay plus the Stage-A legal candidate pool,
with 46,324 Train contexts, 10,080 Moving-Val contexts, and 68,702 legal
candidate samples. Frozen reduced12 ST-GCN features and the frozen shared head
were reused. Train branches used record-balanced sampling (313 records × 16
contexts per epoch), seed 42, CUDA, and 12 epochs; checkpoints are external.

The historical raw pair oracle was verified to be candidate-only in its
ranking (agreement 1.0 with candidate GT-TrueLogP selection). The corrected
normalized pair TrueLogP and pair-margin ceilings were 0.681052/0.674501 and
0.706647/0.701819 Accuracy/Macro-F1, respectively; Pair AnyCorrect coverage
was 7,123/10,080 (0.706647).

Random B2 MeanLogP was 0.429762/0.424594. The best deployable Train-only
branch, Feature + Posterior + Geometry trained on normalized pair TrueLogP,
was 0.517361/0.504616; its best fixed fusion was ConfidenceWeighted at
0.518056/0.510230. Conditional B3 completed: learned B2 + learned B3 was
0.558036/0.548492 versus B2 + random third view 0.533730/0.521124 and the
privileged B3 margin oracle 0.669544/0.665012.

The preregistered result is **KEEP O0-conditioned second-view selection**:
O0 features contribute information (normal versus both-state shuffle differs
by 1.806pp), but complementary candidate utility remains difficult to rank.
The next research decision is whether to improve complementarity
representation before expanding beyond the conditional B3 diagnostic.

Report and implementation:
`experiments/reduced12_eight_placement_v1/overnight_o0_conditioned_second_view/`
and
`activeview/scripts/experiments/run_reduced12_o0_conditioned_second_view.py`.

Task status: **CLEAN**.
