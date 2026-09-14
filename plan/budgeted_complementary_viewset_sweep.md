# Budgeted complementary multi-view sweep

## Scope

- Evaluate reduced12 finite observation budgets B=2, 3, and 4 on 10,080
  Moving-Val contexts.
- Use the exact `current/Stay + Stage-A legal candidate_pool` action set,
  frozen ST-GCN/shared head, and normalized MeanLogP fusion.
- Derive pair and set priors from Train only; keep GT-label set selection as
  a privileged oracle diagnostic.
- Do not read Policy Test, train a model, or generate RGB/skeleton/DINO.

## Checklist

- [x] Reproduction gate for Random B2/B3/B4 and StaticViewPairPrior B2.
- [x] Train pair complementarity matrix with alpha=10 smoothing.
- [x] Geometry, pair-greedy, relative, and navigation-aware policies.
- [x] Train smoothed unordered set3 prior and direct B3 selection.
- [x] Exact B2/B3 oracle and fixed beam-32 B4 reference.
- [x] Rescue, budget, per-class, subset, path-cost, and leakage reports.
- [x] Write result JSON/analysis and verify CUDA runtime.
- [ ] Commit only task files/results and push to `origin/main`.

## Acceptance

The preregistered decision is based on deterministic PairMeanGreedy:
`KEEP` requires B2 >= 0.50 and B3 >= 0.54; `STRONG KEEP` additionally
requires B2 >= 0.51, B3 >= 0.56, and B3 - Random B3 >= 7 pp. If B4-B3 is
under 2 pp, B=3 is the recommended stopping budget. Exact B4 may decrease
because fixed MeanLogP fusion can dilute evidence from an added view.
