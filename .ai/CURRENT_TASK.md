# Budgeted complementary multi-view sweep — completed 2026-09-14

Completed the reduced12 Train/Moving-Val-only Budgeted Complementary
Multi-View Active HAR Sweep. Moving Val has 10,080 contexts and Train has
46,324 contexts. The action set is exactly `current/Stay + Stage-A legal
candidate_pool`; B=2/3/4 adds B-1 candidates and uses fixed normalized
MeanLogP fusion. Policy Test was not read, no model was trained, and no new
RGB/skeleton/DINO artifact was generated.

The registered gate is reproduced exactly: Random B2/B3/B4 are
0.429762/0.424594, 0.487401/0.477211, and 0.521230/0.505236;
StaticViewPairPrior B2 is 0.508234/0.497486 (Accuracy/Macro-F1).
Corrected recursive PairMeanGreedy reaches 0.508234/0.497486 (B2),
0.557639/0.546970 (B3), and 0.569544/0.556675 (B4). SmoothedSet3Prior B3
is 0.550298/0.540837. Exact privileged GT-Margin results are
0.706647/0.701819 (B2), 0.738591/0.737738 (B3), and fixed beam-32 B4
0.721032/0.719061, documenting MeanLogP dilution at B4.

The deterministic decision is **KEEP COMPLEMENTARY SET-SELECTION** (B2 is
below the STRONG KEEP 0.51 gate). PairMeanGreedy gains +20.565 pp, +4.940
pp, and +1.190 pp over Stay for B2/B3/B4; B4 adds <2 pp over B3, so B=3 is
the practical stopping budget. Relative-geometry-only priors underperform
the Train pair prior, and pair-margin correlations with angular/radius
distance are near zero, indicating nontrivial complementarity structure.

Report and implementation:
`experiments/reduced12_eight_placement_v1/budgeted_complementary_viewset_sweep/`
and
`activeview/scripts/experiments/run_reduced12_budgeted_complementary_viewset_sweep.py`.

Task status: **CLEAN pending commit/push**.
