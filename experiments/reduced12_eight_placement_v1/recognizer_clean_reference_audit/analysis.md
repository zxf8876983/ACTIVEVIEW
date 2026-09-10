# Recognizer-Level Clean Motion Reference Audit

## Status: BLOCKED before instance-level analysis

The Val record-to-source audit was run, but the required exact clean H36M17 reference could not be constructed from the repository's current formal pipeline. No clean skeleton, ST-GCN output, similarity, ranking, quadrant, selector, or per-class metric was fabricated.

- Val Moving contexts: 10080
- Unique Val record instances: 105
- Exact record/segment mapping: confirmed
- Exact clean motion instances matched: 0
- Blockers: no explicit clean/true H36M17 skeleton cache was found; formal AMASS-to-H36M17 clean converter is unavailable

## What is available

Every checked Val record carries an AMASS source path, start/end frame, and the deterministic 30-frame `np.linspace` mapping. The archived candidate NPZs are view-dependent estimated skeletons and therefore are not accepted as clean references.

## Why the analysis stopped

`activeview.data.motion.babel_official150_true_skeleton` does not define the referenced `AMASSTrueSkeletonConverter`, and no explicit clean/true H36M17 skeleton cache exists at the checked dataset locations. The installed environment also lacks an alternative formal clean FK path. Generating a clean reference by choosing another same-class motion, using a centroid, or treating an estimated candidate as clean would violate exact instance alignment.

## Required minimum fix

Restore or provide the project's formal AMASS-SMPL/SMPL-X-to-H36M17 FK converter and its exact normalization (`root_center + torso_scale + yaw_only`), or provide a verified per-record clean skeleton cache keyed by the exact `record_id` and frame segment. Then rerun this script before computing recognizer-level comparisons.

Flags: `test_used=false`; `training_used=false`; `new_rgb_rendered=false`; `new_pose_estimation=false`; `exact_clean_motion_instance_required=true`; `clean_motion_used_for_privileged_reference_only=true`; `clean_reference_used_as_deployable_input=false`; `deployable=false`.

`representation_cases.png` was intentionally not generated because the alignment gate failed and no valid cases exist.
