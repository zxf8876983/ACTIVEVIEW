# Stage D — Sequential Active View Selection

Stage C-v0 is frozen as a first-stage Top-3 proposal model. Stage D studies
whether one real intermediate observation allows a second decision without
exposing unvisited candidate perception.

This directory contains controlled Stage D experiment records. EXP014 and
EXP015 have completed their authorized Val-only runs; Test, Habitat rendering
and perception regeneration remain outside the protocol.

## Causal contract

At `t0`, only the current observation and candidate geometry are visible. The
frozen Stage C-v0 Set Ranker proposes Top-3 using its existing utility-descending
and geodesic/viewpoint-ID tie-break. If its maximum predicted utility is not
positive, the episode stays at `s0` and terminates. Otherwise the robot visits
proposal `p1` and only then obtains the cached `s1` perception. EXP014 may use
`s0`, `s1`, their 19-D semantic delta and `s1→p2/p3` geometry. Perception for
unvisited `p2/p3` is never a policy input.

## Navigation cache

No pairwise viewpoint geodesic cache was present during preparation. Before a
future run, `build_pairwise_viewpoint_geodesic.py` must create one JSON matrix
per scene/region using only Habitat Pathfinder shortest-path queries. It must
not create sensors, RGB/depth, humanoids or perception outputs.

## Experiment definitions

- **EXP014** asks whether one observed intermediate state improves final
  recognition beyond the one-shot plateau. It trains only a small
  `SequentialObservationRanker` on Train second-step samples and selects by
  complete Val trajectory Macro-F1.
- **EXP015** performs no training. It compares the fixed-first second-step
  oracle with NoMove, frozen v0, EXP014, CandidateOracle and SafeOracle to
  quantify remaining budget and initial-Stay ceilings.
- **EXP016** is prepared as a no-training, Val-only error decomposition. It
  keeps the frozen v0 first decision fixed and separately replaces the
  second-step gate and p2/p3 candidate identity with offline oracle choices.

Execution outcomes are recorded in each experiment's `result.json` and
`analysis.md`. After the geometry correction and Val-only rerun, EXP014 is
**REJECT** under its recorded thresholds; EXP015 remains an analysis-only
**INCONCLUSIVE** budget/oracle diagnostic. No Test evaluation was performed and
no frozen upstream artifact was changed.

## Geometry-semantics audit status

Post-run review found that the pre-fix Stage D cache populated the sixth and
seventh geometry features with an s1-camera displacement bearing rather than
Stage A's radial `azimuth_deg` difference. The implementation now uses the
existing semantic-region-v2 metadata and the exact Stage A wrapping rule. The
cache was rebuilt and both approved Val-only experiments were rerun. The
pre-fix runtime is archived under the corresponding
`*_pre_geometry_fix` directory and is not used for the corrected decisions.
