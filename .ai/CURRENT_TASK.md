# Static View Prior + Frame0 Residual NBV Audit — completed 2026-09-14

Completed the strict Train/Moving-Val-only Frame0 single-step audit on the
reduced12 eight-placement protocol. The action set was exactly current/Stay
plus the Stage-A legal candidate pool; each selected action was evaluated from
the selected real O1 alone with the frozen ST-GCN and old adaptive head. Policy
Test was not read and no perception artifact was regenerated.

The Train-derived static viewpoint prior reproduced 0.522917 Accuracy /
0.546040 Macro-F1 on 10,080 Moving-Val contexts. A residual selector using
Frame0 RGB DINO + geometry + the existing VisibilityAux architecture reached
0.525496/0.548474 at lambda=1.0; lambda=0.5 reached 0.524008/0.547073 and
geometry-only residual reached 0.518948/0.543900. The existing instance-only
adaptive-aware selector was 0.526984/0.547002.

The best residual branch improved over StaticViewPrior by only +0.258pp
Accuracy and +0.243pp Macro-F1, and was -0.149pp Accuracy below the
instance-only selector. RGB shuffling reduced the residual branch by 1.111pp;
residual-zero exactly reproduced prior actions and metrics. The strict decision
is **KILL PRIOR+RESIDUAL FRAME0 NBV**: Frame0 RGB has measurable residual
signal, but not enough incremental NBV value to justify this route.

Artifacts:
`experiments/reduced12_eight_placement_v1/prior_residual_frame0_nbv/`
and
`activeview/scripts/experiments/run_reduced12_prior_residual_frame0_nbv.py`.

Task status: **CLEAN**; task-owned commit and push completed.
