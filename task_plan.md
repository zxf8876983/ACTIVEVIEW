# Task Plan: RGB-D Human-State Recovery + Deployable Two-Policy Gate

## Goal
Implement and run the Train/Moving-Val RGB-D human-state recovery and two-policy gating audit without reading Policy Test or changing frozen models.

## Phases
- [x] Phase 1: Verify protocol, paths, GPU and causal VideoPose3D behavior
- [x] Phase 2: Build current-frame YOLO/depth compact caches and train-derived proxy
- [x] Phase 3: Evaluate D0/D1/D2/D3 visibility and policy complementarity
- [x] Phase 4: Write reports, validate code, update project records, commit and push

## Key Questions
1. Does the existing frame-0 VideoPose3D estimate read future frames?
2. Can direct current-frame RGB-D state recover enough scene visibility for deployable gating?
3. Which, if any, deployable policy pair and gate improve over D2?

## Decisions Made
- Use only Policy Train and Moving Val; never open Test artifacts.
- D2 will use direct RGB-D backprojection plus a Train-derived canonical template when VideoPose3D is non-causal.
- Raw RGB/depth/large checkpoints remain outside Git; reports contain compact summaries only.

## Errors Encountered
- The first metric pass repeatedly recomputed `select_actions` inside per-row
  comprehensions, making gate evaluation quadratic.  The final runner
  materializes Train/Val actions once before gate analysis.
- The policy-landscape cache stores logits before the frozen Policy-balanced
  head; the final runner correctly uses the Yaw8Fair feature cache and shared
  head to preserve the matched recognizer protocol.

## Status
**Completed** - reports and compact runtime-cache metadata are written under
`experiments/reduced12_eight_placement_v1/rgbd_deployable_two_policy_gate_v1/`.
