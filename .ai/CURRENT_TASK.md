# Known-Map + Current-Depth Movement-Aware NBV — completed

Train/Moving-Val only; no Policy Test, no model training, and no RGB/skeleton/
DINO regeneration. The audit used the frozen reduced12 Yaw8 recognizer and
the exact Stage-A legal candidate-only action set on 10,080 Moving-Val
contexts (68,702 candidates).

Current Frame-0 depth was rendered with Habitat using eight spawned workers.
Raw depth and point clouds were transient; only compact per-candidate path,
clearance and risk summaries were persisted under the runtime data root.
Train was used only for the static prior and a record-holdout lambda choice.

Key results:

- StaticPrior: 0.549901 Accuracy / 0.571990 Macro-F1.
- RGBGlobal-Visibility: 0.567361 / 0.582567; mean/median path 2.7876/2.2245 m.
- RGBGlobal-Visibility + MapPath (lambda=.10): 0.564881 / 0.581486;
  mean/median path 2.6633/2.1775 m (4.46% shorter, -0.248pp Accuracy).
- RGBGlobal-Visibility + MapPath + CurrentDepth: 0.563690 / 0.580532;
  mean/median path 2.5673/1.7881 m; selection switch 5.34%, clearance +5.15%
  relative and depth-risk rate -5.45% relative to MapPath.
- GT SceneVisibility: 0.583929 / 0.598955; GT-TrueLogP Oracle:
  0.760714 / 0.775961.
- Depth occupancy counts were 400,428,767 non-human and 50,413,318
  bbox-attributed points; human-attributable fraction 0.1118 and navmesh
  free-space proxy fraction 0.1625 (deterministic 2,048-point sample/context).

The preregistered 20% distance gate was not met, so PATH-COST NBV is not
useful under this benchmark. Current depth changes 5.34% of selections and
shows mixed but sub-gate local-risk evidence; no automatic follow-up was
started. Reports are in
`experiments/reduced12_eight_placement_v1/movement_aware_depth_nbv/` and the
runner is
`activeview/scripts/experiments/run_reduced12_movement_aware_depth_nbv.py`.
