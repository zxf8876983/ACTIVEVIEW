# RGB-D Human-State Recovery + Deployable Two-Policy Gate — completed

The reduced12 Train/Moving-Val audit is complete. Existing VideoPose3D was
confirmed non-causal for frame 0 (filter widths `[3,3,3,3,3]`, padding 121,
receptive field 243), so deployable D2 uses only current-frame RGB-D
backprojection, current-frame YOLO26n-Pose, and a Train-derived H36M17
template. No Policy Test data, candidate RGB/depth/YOLO, or future frames
entered deployable selection.

Moving Val (10,080 contexts; 68,702 legal candidates) with frozen Yaw8 ST-GCN
plus Policy-balanced shared head:

- D0 privileged Frame0SceneVisibility: 0.583929 Accuracy / 0.598955 Macro-F1
- D1 RGB-D root + GT relative pose: 0.499802 / 0.524925
- D2 strict current RGB-D state: 0.500496 / 0.521328
- D3 uncertainty-weighted visibility (alpha .25/.50): 0.500595 / 0.521707 and
  0.500496 / 0.521462
- Random legal: 0.426786 / 0.450423
- Static prior: 0.523413 / 0.544567

RGB-D depth caches were generated with eight spawned Habitat workers for
current frame only; raw depth was not persisted. Root localization was
numerically exact in the synthetic render (median 0 m); reconstructed
world-joint MPJPE was 0.4098 m (0.2556 m observed and 0.6308 m template
completed). D2 is 8.343 pp below D0, so the preregistered viability gate fails
and human-state recovery is the current bottleneck. Train record-holdout
complementarity selected `A_dep+C` (0.570978 pair AnyCorrect), below the 0.61
gate threshold; no complex learned gate was trained. Results are in
`experiments/reduced12_eight_placement_v1/rgbd_deployable_two_policy_gate_v1/`.
