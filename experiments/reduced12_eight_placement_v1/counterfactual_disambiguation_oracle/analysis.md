# Action-Agnostic Counterfactual Disambiguation Oracle

## Status: STOPPED_PAIRING_INTEGRITY

The main selector was not evaluated. The protocol requires a same-scene, same-placement, same-camera counterfactual for all 12 action hypotheses, including the stay/current viewpoint. Existing Val archives do not provide that strict pairing.

- Moving Val contexts: 10080
- Stay same-camera 12-action coverage: 1154/10080 (11.448%)
- Candidate same-camera 12-action coverage: 68702/68702 (100.000%)
- Scene/placement groups with action-dependent legal-set mismatch: 134

Because a current-viewpoint counterfactual is missing for most contexts, substituting another initial viewpoint or using nearest-neighbor records would violate the requested pairing contract. Therefore no JSD, feature-separability, terminal selector, or regret metric was produced.

The frozen s0 hypothesis-support audit is still valid: see hypothesis_coverage.json. It is not a selector result.

test_used=false; training_used=false; gt_action_used_for_selector=false; predicted_action_used=false; deployable=false.
