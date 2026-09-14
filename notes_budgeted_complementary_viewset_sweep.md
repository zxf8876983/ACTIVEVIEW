# Budgeted complementary multi-view sweep — 2026-09-14

Completed the Train/Moving-Val-only reduced12 sweep with frozen ST-GCN/shared
head and normalized MeanLogP fusion. Moving Val contains 10,080 contexts;
Train contains 46,324 contexts. The action set is exactly current/Stay plus
the Stage-A legal candidate pool. Policy Test and new perception artifacts
were not read or generated.

The reproduction gate matches all registered random baselines:
Random B2 0.429762/0.424594, B3 0.487401/0.477211, B4 0.521230/0.505236,
and StaticViewPairPrior B2 0.508234/0.497486 (Accuracy/Macro-F1).

With the corrected recursive pair prior, PairMeanGreedy obtains
0.508234/0.497486 at B2, 0.557639/0.546970 at B3, and
0.569544/0.556675 at B4. SmoothedSet3Prior reaches 0.550298/0.540837 at
B3. Exact privileged GT-Margin set results are 0.706647/0.701819 (B2),
0.738591/0.737738 (B3), and the fixed beam-32 B4 reference is
0.721032/0.719061; B4 dilution is therefore recorded.

PairMeanGreedy's incremental Accuracy gains over Stay (0.302579) are
+20.565 pp, +4.940 pp, and +1.190 pp for B2/B3/B4. It passes the KEEP gate
but not STRONG KEEP because B2 is below 0.51. PairMeanGreedy exceeds Random
by +7.024 pp at B3, while B4 adds less than 2 pp over B3, supporting B=3 as
the practical budget. Relative-geometry-only B2/B3 (0.434226/0.423513 and
0.493948/0.482568) underperform the Train pair prior; pair complementarity
is not reducible to angular/radius distance (Spearman correlations are near
zero), although the pair matrix is moderately symmetric (Spearman 0.816).

This is a deterministic structure-discovery audit, not a deployable policy;
continuous human/navigation synchronization is not modeled.
