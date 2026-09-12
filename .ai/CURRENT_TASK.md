# Privileged information ladder / oracle gap decomposition — completed 2026-09-13

Implemented and ran a Train/Val-only reduced12 frame-0 information ladder
over 46,324 Policy Train contexts and 10,080 Moving-Val contexts.  The action
set is exactly current/Stay plus the Stage-A legal candidate pool.  The frozen
reduced12 ST-GCN and shared head were reused; no Policy Test, new perception
artifacts, or production model changes were used.

Moving-Val Accuracy/Macro-F1:

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Stay | 0.302579 | 0.292976 |
| Random | 0.365079 | 0.364567 |
| GeometryOnly Utility | 0.487302 | 0.488664 |
| RealVisibility+Geometry | 0.520139 | 0.520467 |
| GTAction+Geometry | 0.488889 | 0.489240 |
| GTAction+RealVisibility+Geometry | 0.526091 | 0.522288 |
| GT-TrueLogP Oracle | 0.753175 | 0.755358 |

The descriptive residual Oracle→GTAction+RealVisibility+Geometry gap is
22.708pp Accuracy.  Geometry plus real visibility provides +3.284pp over the
unified GeometryOnly rerun, while adding GT action to geometry alone provides
only +0.159pp.  The preregistered interpretation is conclusion D: a large
pre-action residual remains even with action and visibility cues, so the
pre-action oracle is not realistically predictable.

Report and implementation:
`experiments/reduced12_eight_placement_v1/privileged_information_ladder/`
and
`activeview/scripts/experiments/run_reduced12_privileged_information_ladder.py`.

Task status: **CLEAN**. No follow-up method was started automatically.
