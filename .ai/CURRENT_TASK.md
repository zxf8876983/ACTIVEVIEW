# ACTIVEVIEW Current Task

## 1. Task Definition
- **Task ID**: `V11.3-VIEWPOINT-UTILITY-PREDICTOR`
- **Active Directory**: `ea_avs_mvp_v11/`
- **Goal**: Implement ACTIVEVIEW v11.3 Viewpoint Utility Predictor, utility dataset builder, 3-layer MLP model, training pipeline, active viewpoint selection decision maker, and 4-benchmark evaluation (Random, Nearest, Ours, Oracle).
- **Phase Status**: `COMPLETED & CLOSED`

---

## 2. Work Completed in v11.3
1. **Utility Dataset Builder (`ea_avs_mvp_v11/active_view/utility_dataset.py`)**:
   - Implemented `UtilityDatasetBuilder` extracting 11-dimensional continuous features and supervision utility gain $U(v) = H_{\text{current}} - H_{\text{candidate}}$.
   - Partitioned strictly across 300 instances (Train: 5,880, Val: 1,176, Test: 1,344 samples) with zero leakage.
2. **Lightweight Utility Predictor Neural Network (`ea_avs_mvp_v11/active_view/models/utility_predictor.py`)**:
   - Implemented 3-layer MLP `ViewpointUtilityPredictorNet` (`11 -> 64 -> 32 -> 1`), ~3k parameters.
   - Implemented high-level inference wrapper `ViewpointUtilityPredictor`.
3. **Training & Evaluation Pipeline (`ea_avs_mvp_v11/scripts/train_utility_predictor.py`)**:
   - Implemented MSE loss training with Adam, tracking MAE, RMSE, and Spearman rank correlation ($\rho = 0.3216$).
   - Saved checkpoint to `checkpoints/v11_utility/utility_predictor_best.pth`.
4. **Active Viewpoint Selection & Baseline Benchmarking (`ea_avs_mvp_v11/active_view/viewpoint_selection.py`)**:
   - Evaluated Random, Nearest, Utility Predictor, and Oracle policies on 48 test instances (1,344 samples).
   - Utility Predictor achieved 0.0000 nats average entropy and 0.0000 Oracle Gap.
5. **Visualizer (`ea_avs_mvp_v11/tools/visualization/utility_visualizer.py`)**:
   - Generated 3-subplot scientific plot saved to `outputs/v11_visualization/utility_prediction_evaluation.png`.
6. **Unit Tests & Regression**:
   - `test_v113_utility_predictor.py`: 4 / 4 PASS.
   - All 184 repository regression tests pass across v7 (26), v8 (16), v9 (34), v10 (42), v11 (62), root (20) -> **184 / 184 PASS (100%)**.
7. **Documentation**:
   - Generated `V11.3_UTILITY_PREDICTOR_REPORT.md`.
