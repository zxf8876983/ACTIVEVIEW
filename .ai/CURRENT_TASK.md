# Selector complementarity + RGB-D deployable NBV audit — completed (Phase A / capability probe)

The strict Frame-0 selector complementarity audit and a matched Habitat depth
capability probe are complete. The deployable RGB-D path remains blocked until
current-frame depth and raw YOLO keypoint artifacts are separately acquired;
the results below preserve the completed known-map audit for traceability.

Moving-Val (10,080 contexts) pair AnyCorrect rates were A+B 0.620933,
A+C 0.633631, A+D 0.640377 and A+E 0.662004, where A is privileged
Frame0SceneVisibility and E is G_full (lambda=.5). The best pair-containing-A
gate therefore clears the preregistered 0.61 diagnostic threshold, while the
Train record holdout selected E as the alternative. The corresponding
GT-TrueLogP policy-pair oracle for A+E is 0.657639 Accuracy / 0.674987
Macro-F1; no gate was trained because the required current RGB-D state was
not available.

Habitat SensorType.DEPTH was successfully probed on 50 Train and 50
Moving-Val current-frame views (256x256, finite metric depth), but no depth
cache or raw frame-0 YOLO cache exists. Consequently root localization,
estimated-pose visibility, uncertainty features and deployable gates are
marked N/A rather than fabricated. Policy Test was not read.

Artifacts: `experiments/reduced12_eight_placement_v1/rgbd_complementarity_gate_audit/`.

---

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
