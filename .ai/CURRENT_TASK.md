# Real Candidate Quality Privileged Audit — completed 2026-09-12

Ran the Val-only reduced12 eight-placement Moving-Val audit on 10,080
contexts. The matched action set is `current/stay + Stage-A legal
candidate_pool`; all terminal predictions use the frozen reduced12 ST-GCN
feature cache plus the frozen shared adapted head. Policy Test was not read,
and no model, perception data or runtime cache was modified.

Action-agnostic privileged selector results (Accuracy / Macro-F1):

- Stay + Shared: 0.302579 / 0.292976
- Random legal + Shared: 0.365079 / 0.364567
- RealPoseConfidence: 0.513194 / 0.513235
- SceneVisibility: 0.521131 / 0.520336
- VisibleJointRatio: 0.521131 / 0.520336
- HumanVisibility: 0.302877 / 0.300860
- TotalVisibility: 0.504365 / 0.507568
- ProjectedHumanArea: 0.483929 / 0.479323
- TemporalMotionRetention: 0.391071 / 0.378826
- Historical Route-1 + Shared: 0.509524 / 0.509806
- GT-TrueLogP Oracle + Shared: 0.753175 / 0.755358
- Legal AnyCorrect candidate-only coverage: 0.771726

SceneVisibility is the best action-agnostic selector. It is +21.855pp above
Stay, +15.605pp above Random, and +1.161pp above the historical Route-1
shared-head replay, while remaining a privileged future-quality diagnostic.
Its candidate-level Spearman with shared GT true-logp is 0.402764 and its
within-context Spearman mean/median are 0.247205 / 0.261905. Existing
scene/human quality artifacts have no current/stay scalar, so occlusion
stratification is explicitly unavailable rather than filled synthetically.

Artifacts and script:

`activeview/scripts/eval/analyze_reduced12_real_candidate_quality_privileged.py`

`experiments/reduced12_eight_placement_v1/real_candidate_quality_privileged_audit/`

Task status: **CLEAN**. Future-quality prediction remains a possible target,
but any deployable method should combine predicted observation quality with
task evidence; no follow-up predictor was trained automatically.
