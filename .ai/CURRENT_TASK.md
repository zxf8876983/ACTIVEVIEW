# Known-map geometric NBV upgrade — completed

The Val-only known-map geometric suite used the frozen Yaw8 ST-GCN plus the
matched Policy-balanced head on 10,080 Moving Val contexts. The exact
candidate-only Stage-A action set and Yaw8Fair cache signatures were checked;
Policy Test and new perception generation were not used. Existing deterministic
map/raycast features were reused (311 dense proxy points per context).

Moving-Val Accuracy/Macro-F1: StaticViewPrior 0.549901/0.571990,
Frame0SceneVisibility 0.583929/0.598955, DenseVisibility
0.583631/0.598359, DOQ-Equal 0.557341/0.577746, G_full λ=.5
0.583532/0.602979, GT-TrueLogP Oracle 0.760714/0.775961 and GT-Margin
0.776190/0.796334. The baseline reproduction gate passed within 0.2pp.
No geometric score exceeded Frame0SceneVisibility or the 60% milestone;
decision: STOP GEOMETRIC SCORE EXPANSION. Current archives expose only a
30-frame viewpoint confidence scalar, so uncertainty-weighted visibility and
estimated-world-state E1/E2 were correctly marked N/A rather than fabricated.

Artifacts: `experiments/reduced12_eight_placement_v1/known_map_geometric_nbv_upgrade/`.
Runtime map/raycast caches remain external under `ACTIVEVIEW_DATA_ROOT`.
