# GT Human-Mask + Depth Root Recovery Ceiling Audit — completed

Train/Moving-Val only; no Policy Test, model training, or new RGB/skeleton/
DINO perception caches. The audit used current frame-0 Habitat depth plus a
perfect semantic OBJECT_ID humanoid mask, an all-finite-mask point-cloud root,
and the frozen Yaw8 recognizer on the Stage-A legal candidate pool.

Key results on 10,080 Moving-Val contexts (68,702 legal candidates):

- StaticPrior: 0.549901 Accuracy / 0.571990 Macro-F1.
- D0 GT-human-state SceneVisibility: 0.583929 / 0.598955.
- Existing old joint-depth-root D1: 0.499802 / 0.524925.
- GTMask RawRoot D1: 0.553671 / 0.572959.
- GTMask Train-calibrated-root D1: 0.556548 / 0.576580.
- Oracle GT-TrueLogP: 0.760714 / 0.775961.

The GTMask calibrated root has median Euclidean error 0.269271 m and P90
3.172190 m (raw 0.394503 m / 3.193952 m); the old joint-depth reference is
0.354935 m / 2.362966 m. Train-only radial offset b=0.272175 m. D0→best D1
drops 2.738 pp, with selected-view agreement 0.884325 (raw) / 0.892956
(calibrated). Only 7,051/10,080 Val contexts had non-empty human-mask point
clouds.

Conclusion: perfect current-frame human segmentation improves over the old
joint-depth root but remains below D0 and fails the preregistered localization
and route gates. Simple RGB-D root localization is killed; do not expand this
route with additional point-cloud heuristics. Reports are in
`experiments/reduced12_eight_placement_v1/gtmask_depth_root_ceiling/` and the
script is
`activeview/scripts/experiments/run_reduced12_gtmask_depth_root_ceiling.py`.
