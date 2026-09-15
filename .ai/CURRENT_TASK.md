# PepperPose confidence–visibility complementarity audit — completed

Implemented and ran `activeview/scripts/eval/analyze_reduced12_pepperpose_visibility_fusion.py`
with CUDA using only the existing reduced12 Policy Train/Moving-Val artifacts.
The true-facing frame-0 confidence table was loaded from the preceding
`pepperpose_frame0_confidence_audit`; angle sanity and confidence landscapes
were not recomputed. Lambda selection used a deterministic 10% Policy-Train
record holdout, while Moving Val remained evaluation-only.

Moving Val (10,080 contexts) results were:

- RGBGlobal-Visibility: 0.567361/0.582567 Accuracy/Macro-F1.
- RGBGlobal-Visibility + confidence (lambda 0.25): 0.567361/0.583573.
- Frame0SceneVisibility: 0.583929/0.598955.
- Frame0SceneVisibility + confidence (lambda 0.25): 0.502579/0.524444.
- GTYaw-PoseConfidencePrior: 0.437996/0.465250.
- GT-TrueLogP Oracle: 0.760714/0.775961.

The RGB fusion Accuracy gain was 0.000 pp and the Frame0Scene fusion gain
was -8.135 pp. On selector disagreements, confidence corrected 666 RGB
errors but lost 1,970 RGB-correct contexts; for Frame0Scene it corrected 539
errors and lost 2,010. The fixed strict lower-tertile high-occlusion subset
contained 3,271 contexts and showed no useful fusion gain. The registered
decision is **KILL ENTIRE PEPPERPOSE BRANCH**.

The Moving policy archive still exposes sequence-level `(32,)` confidence,
not exact future-candidate frame-0 confidence. The audit therefore retains
the prior's explicit proxy limitation. No RGB was generated, no model was
trained or modified, and Policy Test was not read.
