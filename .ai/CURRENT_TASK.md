# Frame0 True-Facing + YOLO Confidence Angle Prior Audit — completed

Implemented and ran `activeview/scripts/experiments/run_reduced12_pepperpose_frame0_confidence_audit.py` with CUDA on the frozen Yaw8Fair option cache. The prior was constructed from the raw-train-derived Yaw8 internal validation split only (245 source records × 8 yaw variants, 1,960 observations); Policy Test was not read and no model/data was regenerated.

Key results:

- Internal frame-0 confidence ranged from 0.713846 at 135° to 0.937018 at 0° (22.317 pp gap).
- True-facing analytic sanity passed; old placement-yaw and true-facing bins differed in 15/20 samples (0.750).
- Random was 0.426786/0.450423, StaticPrior 0.549901/0.571990, GTYaw-PoseConfidencePrior 0.437996/0.465250, RGBGlobal-Visibility 0.567361/0.582567, Frame0SceneVisibility 0.583929/0.598955, and GT-TrueLogP Oracle 0.760714/0.775961 (Accuracy/Macro-F1).
- Candidate-only AnyCorrect Coverage was 0.776190.
- Internal-to-Moving confidence Spearman was 0.809524, but Moving archives expose only sequence-level `(32,)` confidence, so this correlation is proxy-qualified rather than an exact frame-0 test.

The registered decision is `KILL PEPPERPOSE-STYLE ANGLE PRIOR`: the privileged true-facing prior reached only 0.437996 Accuracy (<0.52), and fusion was correctly skipped by the 0.54 gate. Exact Moving frame-0 candidate confidence cannot be audited without forbidden RGB/YOLO regeneration; the report records this limitation explicitly. Artifacts are under `experiments/reduced12_eight_placement_v1/pepperpose_frame0_confidence_audit/`.
