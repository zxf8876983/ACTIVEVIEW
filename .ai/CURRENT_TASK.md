# Pair complementarity generalization audit — completed 2026-09-14

Completed a CUDA Train/Moving-Val-only audit of whether PairMeanGreedy gains
generalize beyond single-view quality and absolute viewpoint IDs. The exact
action set remained `current/Stay + Stage-A legal candidate_pool`; Train
derived priors used 46,324 contexts and Moving Val used 10,080 contexts.
Policy Test was not read, recognizer/perception artifacts were not modified,
and no model was trained.

Protocol reproduction passed: Random B2/B3 are 0.429762/0.424594 and
0.487401/0.477211; PairMeanGreedy B2/B3 are 0.508234/0.497486 and
0.557639/0.546970 (Accuracy/Macro-F1). AdditiveQuality reaches
0.511310/0.500984 (B2) and 0.555456/0.545044 (B3), so PairMean minus
AdditiveQuality is -0.308/+0.218 pp. ResidualPairGreedy is
0.373413/0.363775 (B2) and 0.418452/0.403702 (B3), which is -6.895 pp below
Random B3; the residual does not support a useful standalone interaction
prior.

Pair matrix versus Q(i)+Q(j) has Spearman/Pearson 0.773338/0.771953, with
residual mean/std 0.394839/0.200308. Cyclic prior shifts reduce B2 to
0.487599 (+45°), 0.485119 (+90°), 0.485714 (+135°), and 0.480754 (+180°);
B3 becomes 0.541270/0.544643/0.544841/0.544742. Maximum drops are 2.748 pp
(B2) and 1.637 pp (B3), below the strong absolute-ID shortcut threshold.

Original and Shared recognizer Train pair matrices correlate at
Spearman/Pearson 0.935007/0.928057 (top-10 overlap 0.40). Original-prior →
Shared-recognizer gives 0.509921/0.498460 (B2) and 0.556845/0.549621 (B3);
Shared-prior → Original-recognizer gives 0.425000/0.402581 and
0.461310/0.438192. The main conclusion is that PairMean's practical gain is
mostly single-view quality / dataset prior, with no strong absolute-ID
dependence; genuine residual complementarity is not established. PairMean
B=3 remains a useful empirical baseline because it beats Random B3 by
7.024 pp, but it should not be interpreted as proven stable pair interaction.

Report and implementation:
`experiments/reduced12_eight_placement_v1/pair_complementarity_generalization_audit/`
and
`activeview/scripts/experiments/run_reduced12_pair_complementarity_generalization_audit.py`.

Task status: **CLEAN**.
