# Notes: Static View Prior + Frame0 Residual NBV Audit

## Protocol
- Action set is current/Stay plus Stage-A legal candidate pool from the validated option cache.
- Terminal prediction uses selected real archived option through the frozen old-adaptive classifier head.
- Train contexts form `Q(v)=mean GT-Margin`; Moving Val is the only evaluation population.
- Policy Test, future candidate observations and future candidate recognizer outputs are not selector inputs.

## Reusable artifacts
- `diagnostics/reduced12_dual_route_overnight/{train,val}_options.{npz,json}`
- `datasets/rgb_reduced12_eight_placement_v1/frame0_current` and matching DINO spatial cache
- `diagnostics/frame0_visibility_predictor_v1`
- frozen old adaptive head and historical TaskUtilityPredictor checkpoints

## Planned outputs
`experiments/reduced12_eight_placement_v1/prior_residual_frame0_nbv/` with prior, training, selector, ranking, shuffle, occlusion and leakage JSON plus `result.json` and `analysis.md`.
