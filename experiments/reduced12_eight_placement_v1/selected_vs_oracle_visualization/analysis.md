# Selected-vs-GT-best Failure Case Visualization Audit

Generated 12 Moving-Val cases from the existing Candidate-Conditioned Spatial selections and Real-GTMargin oracle. No Test data were read, no models were trained, and no RGB/skeleton data were regenerated.

- Mean absolute selected-vs-GT-best azimuth difference: 33.75
- Oracle SceneVisibility higher fraction: 0.000
- Oracle HumanObservability higher fraction: 0.000
- Mean GT-margin difference (oracle - selected): 9.5905
- Motion fidelity: `motion_fidelity_gt_alignment_unavailable` (no reliable GT world-space canonical alignment in current archive).
- RGB: archived visited files contain frame_index=15 only; t0/t29 panels are explicitly N/A.
- Scene occupancy map: unavailable; geometry panels show human and candidate positions only.

## Interpretation

This is a 12-case qualitative audit, not a population estimate. The figures should be used to inspect whether visibility, viewing angle, reconstruction distortion, or temporal evidence plausibly explains selected-vs-oracle failures. Missing caches are shown as N/A rather than reconstructed.

## Protocol flags

`test_used=false`; `training_used=false`; `new_rgb_rendered=false`; `new_pose_estimation=false`; `gt_action_used_for_posthoc_diagnostic_only=true`; `gt_margin_used_for_case_selection_and_visualization_only=true`; `selector_remains_unchanged=true`; `deployable=false`.

## RGB availability and targeted rendering

The RGB loader enforces `available_view_mask`; unavailable zero-filled slots are rendered as N/A/gray and are never interpreted as black RGB. Targeted rendering was limited to 12 cases × 2 viewpoints × frames [0, 15, 29].
Targeted renderer status: `BLOCKED_GPU` (PyTorch CUDA is unavailable).
No full-dataset RGB regeneration was performed.

## Additional protocol flags

`full_dataset_rgb_regenerated=false`; `targeted_rgb_rendering_only=true`; `targeted_cases=12`; `targeted_viewpoints_per_case=2`; `targeted_frames=[0,15,29]`.
