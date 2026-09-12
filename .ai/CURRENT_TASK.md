# Frame-0 Causal Observability Audit — completed 2026-09-12

Completed the Val-only reduced12 eight-placement Moving-Val audit over all
10,080 contexts. The matched action set is `current/stay + Stage-A legal
candidate_pool`; frame-0 selection uses only scene-only Habitat raycasts to
frame-0 reconstructed world-space H36M17 joints. Terminal predictions use the
frozen reduced12 ST-GCN feature cache plus the frozen shared adapted head.
Policy Test was not read, and no model, perception data or runtime cache was
modified.

Matched Accuracy/Macro-F1 results:

- Stay: 0.302579 / 0.292976
- Random legal: 0.365079 / 0.364567
- RealPoseConfidence (non-causal reference): 0.513194 / 0.513235
- Frame0SceneVisibility: 0.489782 / 0.485962
- FullTemporalSceneVisibility (six-frame reference): 0.495833 / 0.492420
- Historical Route-1 + shared head: 0.509524 / 0.509806
- GT-TrueLogP Oracle: 0.753175 / 0.755358

Frame0SceneVisibility is +12.470pp Accuracy over Random and reaches the
pre-registered strong causal threshold (>=48% and >=+8pp), so the audit keeps
a strict causal observability route. FullTemporal exceeds Frame-0 by only
0.605pp Accuracy, while frame-0 versus shared GT true-logp has within-context
Spearman mean/median 0.253482/0.272059. Candidate-only AnyCorrect coverage is
0.771726 and Stay-plus-candidate coverage is 0.791964.

Frame-0 pose confidence is unavailable because archives contain only a
30-frame aggregate viewpoint scalar; no per-frame value was fabricated.

Artifacts and script:

`activeview/scripts/eval/analyze_reduced12_frame0_causal_observability.py`

`experiments/reduced12_eight_placement_v1/frame0_causal_observability_audit/`

Task status: **CLEAN**. No future-quality predictor was trained automatically;
the next decision is whether to authorize training a deployable
`(current RGB + candidate geometry) -> frame-0 visibility score` predictor.
