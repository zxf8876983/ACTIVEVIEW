# Handoff

Status: CLEAN

Stage C-v0 and EXP001–EXP013 remain frozen/recorded. Stage D EXP014/EXP015
were rerun Val-only after correcting the cache's relative-azimuth semantics to
Stage A's radial definition. The corrected EXP014 result is REJECT under its
recorded thresholds; EXP015 remains an analysis-only INCONCLUSIVE diagnostic.
The pre-fix runtime outputs are archived for traceability. Await human review
before any follow-up and do not run Test.

The Stage D cache was built with a navigation-only pairwise 32×32 geodesic
matrix per scene/region. The builder used frozen Stage A/B/C-v0 rows and
cached s1 skeletons, reconstructed s1 through the frozen ST-GCN, and never
used unvisited candidate perception as policy input. Runtime results and
provenance are under `ACTIVEVIEW_DATA_ROOT/experiments/stage_d/`; compact
source records are under `experiments/stage_d/`.

No Test, Habitat rendering, RGB, YOLO, VideoPose3D or ST-GCN retraining was
performed during the rerun. The working tree may contain unrelated
pre-existing untracked files; they are not part of the Stage D result.
