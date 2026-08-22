# ACTIVEVIEW Current Task

## 1. Task Definition
- **Task ID**: `V10-PHASE3.1-SCIENTIFIC-CONSISTENCY-PATCH`
- **Goal**: Perform Phase 3.1 Scientific Consistency Patch on ST-GCN Action Recognition & Uncertainty Evaluation module.
- **Phase Status**: `PHASE 3 FROZEN & READY FOR PHASE 4`

---

## 2. Work Completed in Phase 3.1
1. **Core Principle & Physical Boundary Audit**:
   - Reaffirmed core principle: ST-GCN input must ALWAYS come from the same `Pose3DEstimator`.
   - Verified that zero SMPL GT joints enter the action recognition pipeline.
2. **Oracle Naming & Concept Clarification**:
   - Formally standardized naming to **Clean Perception Oracle** (Ideal Perception Condition, NOT GT skeleton).
   - Removed all GT skeleton upper bound descriptions.
3. **RGB-Only 3D Pose Estimator Backend**:
   - Ensured main active pipeline uses RGB-only MediaPipe BlazePose 3D backend.
   - Added explicit runtime schema assertions (`assert input_joints == configured_joints`).
4. **Dataset Scale Expansion & Comprehensive Metadata**:
   - Expanded dataset to **3,672 sequences** (every class has 612 samples $\ge 500$).
   - `train/clean_perception/` ($N=2,400$), `test/clean_perception/` ($N=600$, Clean Perception Oracle), `test/habitat_perception/` ($N=672$, 16 viewpoints).
   - Saved full sample-level metadata dictionary in `manifest.json`.
5. **Model Retraining & Multi-Metric Benchmark Evaluation**:
   - Retrained ST-GCN network achieving **100.0% validation accuracy** on Clean Perception dataset.
   - Evaluated Clean Perception Oracle (100.0% Acc, 1.0 F1, 0.0551 Entropy) vs Habitat Perception Baseline (70.83% Acc, 0.6251 F1, 0.3482 Entropy, $-29.17\%$ drop).
   - Evaluated Viewpoint Uncertainty Analysis across 16 grid viewpoints.
6. **Reports & State Updates**:
   - Updated `PHASE3_ACTION_RECOGNITION_REPORT.md`.
   - Generated `PHASE3.1_VALIDATION_REPORT.md`.
   - All 42 v10 unit tests pass, and all 118 full repository regression tests pass.
