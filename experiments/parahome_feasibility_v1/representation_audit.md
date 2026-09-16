# Representation audit

ParaHome body_joint_orientations are 6D rotations (first two matrix columns), body_global_transform is a 4x4 body-to-world transform, and joint_positions contains 73 global positions. The optional smplx_pose stores axis-angle radians for 21 body joints, root orientation, translation and 30 hand joints. The existing ACTIVEVIEW MotionConverter expects the same SMPL-X axis-angle convention but a padded 162-value pose vector; the audit uses a thin deterministic adapter and retains hand rotations.

ParaHome coordinates are not numerically identical to the converter's Habitat frame. A single Kabsch rigid transform is fitted from ParaHome body-root trajectories to converted Habitat root trajectories and applied to rigid-object transforms, preserving human-object relative geometry. Per-sequence metrics and ten-frame distance errors are in `representation_audit.json`.

The selected replay uses the existing `scene_id=NONE` clean floor and male_0 URDF. ParaHome gender metadata is retained in `sequence_audit.json`; a production replay should add a gender-specific asset or document the male_0 approximation.
