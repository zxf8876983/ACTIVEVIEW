# Notes: RGB-D human-state recovery audit

## Scope

This audit is restricted to reduced12 eight-placement Policy Train and
Moving Val. Policy Test is locked. D0/D1 are privileged diagnostics; D2/D3
and deployable gates may use only current-frame RGB-D, current YOLO, static
map, candidate geometry and Train-derived templates/statistics.

## Initial evidence

- Existing VideoPose3D uses temporal filter widths `[3, 3, 3, 3, 3]`,
  padding 121 and receptive field 243; frame-0 inference is therefore not
  strict causal and cannot be used by D2.
- Habitat depth sensor was previously probed successfully at 256x256 metric
  depth, but no persistent current-frame depth or YOLO cache existed.
- Existing Yaw8Fair, frame-0 map visibility and candidate metadata caches are
  available under ACTIVEVIEW_DATA_ROOT and are reused read-only.

## Implementation notes

The new runner will create compact current-view caches only, perform exact
YOLO26n-Pose on current RGB, render transient Habitat depth for current frame
0, backproject reliable keypoints, fill missing joints from a Train-derived
template, and run D0/D1/D2/D3 plus Train-holdout policy/gate diagnostics.

## Final run

The run completed on CUDA with eight spawned Habitat workers.  Moving-Val
results were D0 0.583929/0.598955, D1 0.499802/0.524925, D2
0.500496/0.521328 and best D3 0.500595/0.521707 Accuracy/Macro-F1.  The
strict RGB-D path therefore failed the 2pp viability gate and remains a
diagnostic rather than an accepted deployable state estimator.  A quadratic
gate-evaluation loop was fixed by materializing selected actions once; the
final matched Yaw8Fair loader also avoids the raw-logit policy-landscape cache.
