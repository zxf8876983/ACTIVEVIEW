# Handoff

Status: CLEAN

Stage C-v0 and EXP001–EXP013 remain frozen/recorded. The repository now
contains preparation-only Stage D EXP014/EXP015 code and experiment records.
No Stage D training or Val evaluation was run; Test, Habitat, YOLO,
VideoPose3D and ST-GCN retraining were not run. Await human review and explicit
authorization before executing either experiment.

The Stage D cache requires a navigation-only pairwise 32×32 geodesic matrix
per scene/region. The builder uses frozen Stage A/B/C-v0 rows and cached s1
skeletons, reconstructs s1 through the frozen ST-GCN, and never uses unvisited
candidate perception as policy input.
