# RGB-D Coordinate-System & D1 Sanity Audit — completed

Train/Moving-Val only; no Policy Test, model training, recognizer changes, or
new RGB/skeleton/DINO caches. The audit is in
`experiments/reduced12_eight_placement_v1/rgbd_coordinate_sanity_audit/`.

Key findings:

- StaticPrior protocol is repaired: 0.549901 Accuracy / 0.571990 Macro-F1,
  matching the historical Train-derived prior; the earlier 0.523413 result
  used a Val-derived prior inside `_policy_scores`.
- Camera projection/backprojection closes at machine precision (100×17), with
  Habitat world +Y, camera forward −Z, WXYZ camera→world rotation and one
  1.10 m sensor-height addition.
- Independent GT H36M17 root localization has median 0.354935 m and P90
  2.362965 m; the old near-zero root metric was self-consistency, not GT error.
- Synthetic yaw H0–H6 identifies a fixed +90° lateral-axis offset; H2 (−90°)
  gives 0.472° median / 0.473° max error and passes the synthetic gate.
- Error ladder (fixed 256-context depth sample): L0 0 m, L1 GT-pixel + Habitat
  depth 0.9627 m (visible 0.3887 m, occluded 1.6291 m), L2A YOLO-pixel + GT
  depth 0.1572 m, L2B YOLO-pixel + Habitat depth 0.4367 m.
- Full Moving-Val D1 corrected remains 0.499802 / 0.524925 and D2c remains
  0.500496 / 0.521328; D1a-vs-D1b difference is <1e−6 m, so D1 is strictly
  translation-only and the independent root error is the main D1 concern.

The large D2 component gap appears when replacing estimated orientation with GT
orientation on the fixed diagnostic subset; future work should first inspect
pose orientation and depth/surface recovery rather than train a new gate.
